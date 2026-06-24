/*
 * main.cc
 * =======
 * edgecv_model_probe — ESP32-CAM / ESP-IDF v5.3
 *
 * PURPOSE:
 *   The very first probe in this project. Answers one question only:
 *   "Can a 2.9 MB MobileNetV2 TFLite model be loaded from SD card into
 *    PSRAM, and can AllocateTensors() succeed on this hardware?"
 *
 * WHAT IT DOES:
 *   1. Mount SD card (SDMMC 1-bit native mode)
 *   2. Open model.tflite, report file size
 *   3. Allocate a PSRAM buffer sized to the file
 *   4. Read the entire model into that buffer
 *   5. Parse it with tflite::GetModel(), verify schema version
 *   6. Build a MicroMutableOpResolver with the ops MobileNetV2 needs
 *   7. Allocate a 1 MB tensor arena in PSRAM
 *   8. Construct a MicroInterpreter
 *   9. Call AllocateTensors()
 *  10. Print heap/PSRAM statistics throughout
 *
 * DOES NOT DO:
 *   Camera, WiFi, Invoke(), or anything beyond proving the model fits
 *   and allocates correctly in memory.
 *
 * MEASURED RESULT (first successful run):
 *   PSRAM free after AllocateTensors : ~204 KB
 *   Tensor arena allocated           : 1024 KB
 *   Actual arena usage               : ~414 KB
 *   AllocateTensors() time           : ~149 ms
 */

#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_psram.h"

#include "driver/sdmmc_host.h"
#include "driver/gpio.h"
#include "sdmmc_cmd.h"
#include "esp_vfs_fat.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* TAG = "MODEL_PROBE";

#define SD_MOUNT_PT        "/sdcard"
#define MODEL_PATH         "/sdcard/model.tflite"
#define TENSOR_ARENA_SIZE  (1 * 1024 * 1024)   /* 1 MB */

// ---------------------------------------------------------------------------
// Memory statistics helper
// ---------------------------------------------------------------------------

static void print_mem(const char* label)
{
    size_t heap_free     = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t heap_largest  = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
    size_t psram_free    = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t psram_largest = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);

    ESP_LOGI(TAG, "--- MEM [%s] ---", label);
    ESP_LOGI(TAG, "  Internal heap  free: %6zu KB   largest block: %6zu KB",
             heap_free / 1024, heap_largest / 1024);
    ESP_LOGI(TAG, "  PSRAM          free: %6zu KB   largest block: %6zu KB",
             psram_free / 1024, psram_largest / 1024);
}

// ---------------------------------------------------------------------------
// SD card mount — SDMMC 1-bit native mode
//
// AI Thinker ESP32-CAM wiring:
//   CLK -> GPIO 14   CMD -> GPIO 15   D0 -> GPIO 2 (shared with onboard LED)
// ---------------------------------------------------------------------------

static sdmmc_card_t* s_card = nullptr;

static esp_err_t sd_mount(void)
{
    ESP_LOGI(TAG, "Mounting SD card (SDMMC 1-bit native)...");

    gpio_set_direction(GPIO_NUM_2, GPIO_MODE_INPUT);
    gpio_set_pull_mode(GPIO_NUM_2, GPIO_PULLUP_ONLY);
    vTaskDelay(pdMS_TO_TICKS(20));

    esp_vfs_fat_sdmmc_mount_config_t mount_cfg = {
        .format_if_mount_failed = false,
        .max_files              = 5,
        .allocation_unit_size   = 16 * 1024,
    };

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.flags        = SDMMC_HOST_FLAG_1BIT;
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;

    sdmmc_slot_config_t slot_cfg = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_cfg.width   = 1;
    slot_cfg.gpio_cd = SDMMC_SLOT_NO_CD;
    slot_cfg.gpio_wp = SDMMC_SLOT_NO_WP;

    esp_err_t ret = esp_vfs_fat_sdmmc_mount(
        SD_MOUNT_PT, &host, &slot_cfg, &mount_cfg, &s_card);

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SD mount failed: %s", esp_err_to_name(ret));
        return ret;
    }

    sdmmc_card_print_info(stdout, s_card);
    ESP_LOGI(TAG, "SD card mounted at %s", SD_MOUNT_PT);
    return ESP_OK;
}

// ---------------------------------------------------------------------------
// app_main
// ---------------------------------------------------------------------------

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "EdgeCV — Model Load + AllocateTensors Probe");
    ESP_LOGI(TAG, "Model path : %s", MODEL_PATH);
    ESP_LOGI(TAG, "Arena      : %u KB in PSRAM", TENSOR_ARENA_SIZE / 1024);
    ESP_LOGI(TAG, "========================================");

    print_mem("startup");

    if (sd_mount() != ESP_OK) {
        ESP_LOGE(TAG, "ABORT: SD card mount failed.");
        return;
    }
    print_mem("after SD mount");

    // -------------------------------------------------------------------
    // Open model file
    // -------------------------------------------------------------------
    FILE* f = fopen(MODEL_PATH, "rb");
    if (!f) {
        ESP_LOGE(TAG, "ABORT: Cannot open %s", MODEL_PATH);
        return;
    }

    fseek(f, 0, SEEK_END);
    long model_file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    ESP_LOGI(TAG, "Model file size: %ld bytes (%.2f MB)",
             model_file_size, (float)model_file_size / (1024.0f * 1024.0f));

    if (model_file_size <= 0) {
        ESP_LOGE(TAG, "ABORT: Invalid file size.");
        fclose(f);
        return;
    }

    // -------------------------------------------------------------------
    // Allocate PSRAM buffer for model
    // -------------------------------------------------------------------
    uint8_t* model_buf = (uint8_t*)heap_caps_malloc(
        (size_t)model_file_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    if (!model_buf) {
        ESP_LOGE(TAG, "ABORT: PSRAM malloc failed for model buffer (%ld bytes).",
                 model_file_size);
        fclose(f);
        return;
    }
    ESP_LOGI(TAG, "Model buffer allocated at %p", (void*)model_buf);
    print_mem("after model alloc");

    // -------------------------------------------------------------------
    // Load model from SD
    // -------------------------------------------------------------------
    ESP_LOGI(TAG, "Loading model from SD card...");
    int64_t t_load_start = esp_timer_get_time();
    size_t bytes_read = fread(model_buf, 1, (size_t)model_file_size, f);
    fclose(f);
    int64_t t_load_end = esp_timer_get_time();
    float load_ms = (t_load_end - t_load_start) / 1000.0f;

    if ((long)bytes_read != model_file_size) {
        ESP_LOGE(TAG, "ABORT: Read %zu bytes, expected %ld.", bytes_read, model_file_size);
        heap_caps_free(model_buf);
        return;
    }
    ESP_LOGI(TAG, "Model loaded: %zu bytes in %.1f ms", bytes_read, load_ms);
    print_mem("after model load");

    // -------------------------------------------------------------------
    // Parse flatbuffer
    // -------------------------------------------------------------------
    const tflite::Model* model = tflite::GetModel(model_buf);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "ABORT: Model schema version %lu != runtime %d.",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        heap_caps_free(model_buf);
        return;
    }
    ESP_LOGI(TAG, "Model schema version: %lu (OK)", (unsigned long)model->version());

    // -------------------------------------------------------------------
    // Op resolver — MobileNetV2 ops
    // -------------------------------------------------------------------
    static tflite::MicroMutableOpResolver<14> resolver;
    resolver.AddSub();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddRelu6();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddAveragePool2D();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddPad();
    resolver.AddMean();

    // -------------------------------------------------------------------
    // Allocate tensor arena
    // -------------------------------------------------------------------
    uint8_t* tensor_arena = (uint8_t*)heap_caps_malloc(
        TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    if (!tensor_arena) {
        ESP_LOGE(TAG, "ABORT: PSRAM malloc failed for tensor arena (%u KB).",
                 TENSOR_ARENA_SIZE / 1024);
        heap_caps_free(model_buf);
        return;
    }
    ESP_LOGI(TAG, "Tensor arena allocated at %p (%u KB)", (void*)tensor_arena,
             TENSOR_ARENA_SIZE / 1024);

    // -------------------------------------------------------------------
    // Construct MicroInterpreter
    // -------------------------------------------------------------------
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    // -------------------------------------------------------------------
    // AllocateTensors()
    // -------------------------------------------------------------------
    ESP_LOGI(TAG, "Calling AllocateTensors()...");
    int64_t t_alloc_start = esp_timer_get_time();
    TfLiteStatus alloc_status = interpreter.AllocateTensors();
    int64_t t_alloc_end = esp_timer_get_time();
    float alloc_ms = (t_alloc_end - t_alloc_start) / 1000.0f;

    if (alloc_status != kTfLiteOk) {
        ESP_LOGE(TAG, "ABORT: AllocateTensors() FAILED (%.1f ms).", alloc_ms);
        heap_caps_free(tensor_arena);
        heap_caps_free(model_buf);
        return;
    }
    ESP_LOGI(TAG, "AllocateTensors() OK in %.1f ms", alloc_ms);
    ESP_LOGI(TAG, "Arena used: %zu KB of %u KB (%.1f%%)",
             interpreter.arena_used_bytes() / 1024,
             TENSOR_ARENA_SIZE / 1024,
             100.0f * interpreter.arena_used_bytes() / TENSOR_ARENA_SIZE);

    print_mem("after AllocateTensors — FINAL");

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "RESULT: PASS");
    ESP_LOGI(TAG, "  Model loaded from SD into PSRAM.");
    ESP_LOGI(TAG, "  AllocateTensors() succeeded.");
    ESP_LOGI(TAG, "  Ready for camera probe (Stage 2).");
    ESP_LOGI(TAG, "========================================");

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
