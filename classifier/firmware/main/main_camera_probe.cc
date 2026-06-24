/*
 * main.cc
 * =======
 * edgecv_inference — ESP32-CAM (AI Thinker / OV3660) / ESP-IDF v5.3
 *
 * OV3660 PIXEL FORMAT CONSTRAINT:
 *   The OV3660 does NOT support PIXFORMAT_RGB888 through the esp32-camera
 *   driver.  Requesting it returns ESP_FAIL ("Requested format is not
 *   supported").  Supported formats: JPEG, YUV422, GRAYSCALE.
 *
 *   This application captures YUV422 (YUYV byte order) and converts to
 *   normalised float32 RGB inside fill_input_tensor(), matching the
 *   training pipeline exactly.
 *
 * YUV422 (YUYV) layout — 2 bytes per pixel, 4 bytes per 2-pixel block:
 *   byte 0: Y0   (luma for pixel 0)
 *   byte 1: U0   (blue-difference chroma, shared by pixels 0 and 1)
 *   byte 2: Y1   (luma for pixel 1)
 *   byte 3: V0   (red-difference chroma, shared by pixels 0 and 1)
 *   ...
 *   Total frame size: 96 * 96 * 2 = 18 432 bytes
 *
 * YUV -> RGB conversion (BT.601 full-range, matches PIL/OpenCV defaults):
 *   R = Y + 1.402   * (V - 128)
 *   G = Y - 0.34414 * (U - 128) - 0.71414 * (V - 128)
 *   B = Y + 1.772   * (U - 128)
 *   Values clamped to [0, 255] before normalising to [0.0, 1.0].
 *
 * This matches what the training script produces:
 *   img = Image.open(path).convert("RGB")   <- PIL uses BT.601 for YCbCr->RGB
 *   arr = np.array(img, dtype=np.float32) / 255.0
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

#include "esp_camera.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

static const char* TAG = "INFERENCE";

#define SD_MOUNT_PT           "/sdcard"
#define MODEL_PATH            "/sdcard/model.tflite"

#define INPUT_W               96
#define INPUT_H               96
#define INPUT_C               3

#define TENSOR_ARENA_SIZE     (1 * 1024 * 1024)

#define INFERENCE_INTERVAL_MS 5000

static const char* CLASS_LABELS[] = { "guava", "powerbank" };
static const int   NUM_CLASSES    = 2;

/* AI Thinker OV3660 hard-wired pins */
#define CAM_PIN_PWDN    32
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK     0
#define CAM_PIN_SIOD    26
#define CAM_PIN_SIOC    27
#define CAM_PIN_D7      35
#define CAM_PIN_D6      34
#define CAM_PIN_D5      39
#define CAM_PIN_D4      36
#define CAM_PIN_D3      21
#define CAM_PIN_D2      19
#define CAM_PIN_D1      18
#define CAM_PIN_D0       5
#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline int clamp255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

static void print_mem(const char* label)
{
    ESP_LOGI(TAG, "MEM [%s]  heap=%zu KB  psram=%zu KB", label,
             heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024,
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM)   / 1024);
}

// ---------------------------------------------------------------------------
// SD card
// ---------------------------------------------------------------------------

static sdmmc_card_t* s_card = nullptr;

static esp_err_t sd_mount(void)
{
    gpio_set_direction(GPIO_NUM_2, GPIO_MODE_INPUT);
    gpio_set_pull_mode(GPIO_NUM_2, GPIO_PULLUP_ONLY);
    vTaskDelay(pdMS_TO_TICKS(20));

    esp_vfs_fat_sdmmc_mount_config_t mcfg = {
        .format_if_mount_failed = false,
        .max_files              = 5,
        .allocation_unit_size   = 16 * 1024,
    };
    sdmmc_host_t host   = SDMMC_HOST_DEFAULT();
    host.flags          = SDMMC_HOST_FLAG_1BIT;
    host.max_freq_khz   = SDMMC_FREQ_HIGHSPEED;

    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width   = 1;
    slot.gpio_cd = SDMMC_SLOT_NO_CD;
    slot.gpio_wp = SDMMC_SLOT_NO_WP;

    esp_err_t ret = esp_vfs_fat_sdmmc_mount(SD_MOUNT_PT, &host, &slot,
                                             &mcfg, &s_card);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SD mount failed: %s", esp_err_to_name(ret));
        return ret;
    }
    sdmmc_card_print_info(stdout, s_card);
    return ESP_OK;
}

// ---------------------------------------------------------------------------
// Camera — YUV422, the only colour format OV3660 supports
// ---------------------------------------------------------------------------

static esp_err_t camera_init(void)
{
    camera_config_t cfg = {};
    cfg.pin_pwdn      = CAM_PIN_PWDN;
    cfg.pin_reset     = CAM_PIN_RESET;
    cfg.pin_xclk      = CAM_PIN_XCLK;
    cfg.pin_sccb_sda  = CAM_PIN_SIOD;
    cfg.pin_sccb_scl  = CAM_PIN_SIOC;
    cfg.pin_d7 = CAM_PIN_D7; cfg.pin_d6 = CAM_PIN_D6;
    cfg.pin_d5 = CAM_PIN_D5; cfg.pin_d4 = CAM_PIN_D4;
    cfg.pin_d3 = CAM_PIN_D3; cfg.pin_d2 = CAM_PIN_D2;
    cfg.pin_d1 = CAM_PIN_D1; cfg.pin_d0 = CAM_PIN_D0;
    cfg.pin_vsync    = CAM_PIN_VSYNC;
    cfg.pin_href     = CAM_PIN_HREF;
    cfg.pin_pclk     = CAM_PIN_PCLK;

    cfg.xclk_freq_hz = 20000000;          /* OV3660 requires >=20 MHz */
    cfg.ledc_timer   = LEDC_TIMER_0;
    cfg.ledc_channel = LEDC_CHANNEL_0;

    /*
     * PIXFORMAT_YUV422: only colour format the OV3660 driver supports.
     * Frame buffer: 96 * 96 * 2 = 18 432 bytes in PSRAM.
     * Pixel order: YUYV (Y0 U0 Y1 V0 repeating).
     */
    cfg.pixel_format = PIXFORMAT_YUV422;
    cfg.frame_size   = FRAMESIZE_96X96;
    cfg.jpeg_quality = 12;
    cfg.fb_count     = 1;
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

    ESP_LOGI(TAG, "Camera: YUV422 96x96 fb_count=1 PSRAM xclk=20MHz");
    esp_err_t ret = esp_camera_init(&cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_camera_init failed: %s", esp_err_to_name(ret));
    }
    return ret;
}

// ---------------------------------------------------------------------------
// fill_input_tensor
//
// Converts YUYV frame buffer -> float32 RGB tensor normalised to [0,1].
//
// YUYV layout (4 bytes = 2 pixels):
//   src[0]=Y0  src[1]=U  src[2]=Y1  src[3]=V
//
// BT.601 full-range coefficients (same as PIL Image.convert("RGB")):
//   R = Y + 1.402   * (V-128)
//   G = Y - 0.34414 * (U-128) - 0.71414 * (V-128)
//   B = Y + 1.772   * (U-128)
//
// Output tensor layout: [R,G,B, R,G,B, ...] row-major, float32 /255.
// ---------------------------------------------------------------------------

static void fill_input_tensor(const camera_fb_t* fb, TfLiteTensor* input)
{
    const uint8_t* src = fb->buf;   /* YUYV, 2 bytes/pixel */
    float*         dst = input->data.f;
    const int      npix = INPUT_W * INPUT_H;   /* 9216 pixels */

    for (int i = 0; i < npix; i += 2) {
        /* One 4-byte block covers pixels i and i+1 */
        int Y0 = src[0];
        int U  = src[1];
        int Y1 = src[2];
        int V  = src[3];
        src += 4;

        int u128 = U - 128;
        int v128 = V - 128;

        /* Pixel i */
        int r0 = clamp255(Y0 + ((179 * v128) >> 7));           /* 1.402   ≈ 179/128 */
        int g0 = clamp255(Y0 - ((44  * u128 + 91 * v128) >> 7));/* 0.34414 ≈ 44/128, 0.71414 ≈ 91/128 */
        int b0 = clamp255(Y0 + ((227 * u128) >> 7));           /* 1.772   ≈ 227/128 */

        /* Pixel i+1 (same U,V) */
        int r1 = clamp255(Y1 + ((179 * v128) >> 7));
        int g1 = clamp255(Y1 - ((44  * u128 + 91 * v128) >> 7));
        int b1 = clamp255(Y1 + ((227 * u128) >> 7));

        /* Write to float32 tensor, normalised /255 */
        const float inv255 = 1.0f / 255.0f;
        *dst++ = r0 * inv255;
        *dst++ = g0 * inv255;
        *dst++ = b0 * inv255;
        *dst++ = r1 * inv255;
        *dst++ = g1 * inv255;
        *dst++ = b1 * inv255;
    }
}

// ---------------------------------------------------------------------------
// app_main
// ---------------------------------------------------------------------------

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "EdgeCV — End-to-End Inference (OV3660)");
    ESP_LOGI(TAG, "  Capture : YUV422 96x96 -> RGB float32");
    ESP_LOGI(TAG, "  Norm    : / 255.0  (matches training)");
    ESP_LOGI(TAG, "  Classes : [0]=%s  [1]=%s",
             CLASS_LABELS[0], CLASS_LABELS[1]);
    ESP_LOGI(TAG, "========================================");
    print_mem("startup");

    // SD
    if (sd_mount() != ESP_OK) { return; }
    print_mem("after SD");

    // Load model
    FILE* f = fopen(MODEL_PATH, "rb");
    if (!f) { ESP_LOGE(TAG, "ABORT: cannot open %s", MODEL_PATH); return; }
    fseek(f, 0, SEEK_END);
    long model_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t* model_buf = (uint8_t*)heap_caps_malloc(
        (size_t)model_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!model_buf) { ESP_LOGE(TAG, "ABORT: model malloc"); fclose(f); return; }

    if ((long)fread(model_buf, 1, (size_t)model_size, f) != model_size) {
        ESP_LOGE(TAG, "ABORT: model read"); fclose(f); heap_caps_free(model_buf); return;
    }
    fclose(f);
    ESP_LOGI(TAG, "Model: %ld bytes (%.2f MB)", model_size, model_size/1048576.0f);
    print_mem("after model load");

    // Parse + resolver + arena + interpreter (identical to proven probes)
    const tflite::Model* model = tflite::GetModel(model_buf);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "ABORT: schema mismatch"); heap_caps_free(model_buf); return;
    }

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

    uint8_t* tensor_arena = (uint8_t*)heap_caps_malloc(
        TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "ABORT: arena malloc"); heap_caps_free(model_buf); return;
    }

    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "ABORT: AllocateTensors failed");
        heap_caps_free(tensor_arena); heap_caps_free(model_buf); return;
    }
    ESP_LOGI(TAG, "AllocateTensors OK — arena used %zu KB",
             interpreter.arena_used_bytes() / 1024);

    // Validate input tensor
    TfLiteTensor* input = interpreter.input(0);
    if (input->dims->data[1] != INPUT_H ||
        input->dims->data[2] != INPUT_W ||
        input->dims->data[3] != INPUT_C ||
        input->type != kTfLiteFloat32) {
        ESP_LOGE(TAG, "ABORT: input tensor mismatch — got [%d,%d,%d,%d] type=%d",
                 input->dims->data[0], input->dims->data[1],
                 input->dims->data[2], input->dims->data[3],
                 (int)input->type);
        heap_caps_free(tensor_arena); heap_caps_free(model_buf); return;
    }
    ESP_LOGI(TAG, "Input tensor [1,%d,%d,%d] float32 OK", INPUT_H, INPUT_W, INPUT_C);
    print_mem("after AllocateTensors");

    // Camera
    if (camera_init() != ESP_OK) {
        heap_caps_free(tensor_arena); heap_caps_free(model_buf); return;
    }
    print_mem("after camera init");

    // Warm-up
    ESP_LOGI(TAG, "Discarding 3 warm-up frames...");
    for (int i = 0; i < 3; i++) {
        camera_fb_t* wb = esp_camera_fb_get();
        if (wb) esp_camera_fb_return(wb);
        vTaskDelay(pdMS_TO_TICKS(120));
    }

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Inference loop — every %d ms", INFERENCE_INTERVAL_MS);
    ESP_LOGI(TAG, "========================================");

    const size_t expected_fb_len = INPUT_W * INPUT_H * 2;  /* YUV422: 2 bytes/px */
    uint32_t frame_count = 0;

    while (true) {
        int64_t t_start = esp_timer_get_time();

        // Capture
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "[%lu] fb_get returned NULL", (unsigned long)frame_count);
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
            continue;
        }
        if (fb->len != expected_fb_len) {
            ESP_LOGW(TAG, "[%lu] fb->len=%zu expected=%zu",
                     (unsigned long)frame_count, fb->len, expected_fb_len);
            esp_camera_fb_return(fb);
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
            continue;
        }

        // Preprocess: YUYV -> float32 RGB
        int64_t t_pre = esp_timer_get_time();
        fill_input_tensor(fb, input);
        float pre_ms = (esp_timer_get_time() - t_pre) / 1000.0f;

        // Return frame buffer before the long Invoke() — frees DMA memory
        esp_camera_fb_return(fb);

        // Invoke
        int64_t t_inv = esp_timer_get_time();
        if (interpreter.Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "[%lu] Invoke() failed", (unsigned long)frame_count);
            frame_count++;
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
            continue;
        }
        float invoke_ms = (esp_timer_get_time() - t_inv) / 1000.0f;

        // Results
        TfLiteTensor* output = interpreter.output(0);
        float scores[NUM_CLASSES];
        int   best_idx   = 0;
        float best_score = -1.0f;
        for (int i = 0; i < NUM_CLASSES; i++) {
            scores[i] = output->data.f[i];
            if (scores[i] > best_score) { best_score = scores[i]; best_idx = i; }
        }

        float total_ms = (esp_timer_get_time() - t_start) / 1000.0f;

        ESP_LOGI(TAG, "--- Frame %lu ---", (unsigned long)frame_count);
        for (int i = 0; i < NUM_CLASSES; i++) {
            ESP_LOGI(TAG, "  output[%d] %-12s : %.4f  (%.1f%%)",
                     i, CLASS_LABELS[i], scores[i], scores[i] * 100.0f);
        }
        ESP_LOGI(TAG, "  PREDICTION : [%d] %s  (%.1f%%)",
                 best_idx, CLASS_LABELS[best_idx], best_score * 100.0f);
        ESP_LOGI(TAG, "  TIMING     : preprocess=%.1f ms  invoke=%.1f ms  total=%.1f ms",
                 pre_ms, invoke_ms, total_ms);

        frame_count++;

        int64_t sleep_ms = INFERENCE_INTERVAL_MS -
                           ((esp_timer_get_time() - t_start) / 1000);
        if (sleep_ms > 0) vTaskDelay(pdMS_TO_TICKS(sleep_ms));
    }
}
