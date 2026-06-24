/*
 * main_tinyml_probe.cc
 * ====================
 * EdgeCV — TinyML inference on ESP32-CAM (AI Thinker / OV3660)
 * ESP-IDF v5.3 — model embedded via model_data.h / model_data.cc
 *
 * MODEL: Scratch TinyML CNN, INT8 quantized.
 *   Input:  int8[1, 96, 96, 3]  — NOT float32
 *   Output: int8[1, 1]          — single sigmoid neuron, dequantized to [0,1]
 *   Classes (alphabetical): [0] guava, [1] powerbank
 *   output >= 0.5 -> powerbank, output < 0.5 -> guava
 *
 * CRITICAL — INT8 INPUT PIPELINE:
 *   Training: model has layers.Rescaling(1/255) INSIDE it, so the Keras
 *   model saw float [0,255] and normalised internally.
 *   After int8 quantization: the TFLite converter maps the Rescaling layer's
 *   output range [0.0, 1.0] to int8 [-128, 127].
 *   The quantization parameters stored in the input tensor tell us how:
 *     scale      ≈ 1/255  (each raw uint8 step = one int8 step)
 *     zero_point = -128   (uint8=0 maps to int8=-128)
 *   Therefore:  int8_val = (int)(uint8_pixel) - 128
 *   This is equivalent to: int8_val = uint8_pixel + zero_point
 *   (zero_point is -128, so uint8 + (-128) = uint8 - 128)
 *
 *   DO NOT divide by 255 or use float math for the int8 model input.
 *   The Rescaling layer is baked into the quantized weights — applying
 *   it again would double-normalise and produce nonsense.
 *
 * OV3660: capture YUV422, convert Y/U/V -> R/G/B per pixel (BT.601),
 *         then map uint8 R/G/B to int8 by subtracting 128.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#include "model_data.h"
#include "esp_camera.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* TAG = "TINYML";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define INPUT_W               96
#define INPUT_H               96
#define INPUT_C               3
#define TENSOR_ARENA_SIZE     (256 * 1024)   /* 256 KB — TinyML CNN is much
                                                smaller than MobileNetV2;
                                                increase if AllocateTensors
                                                fails */
#define INFERENCE_INTERVAL_MS 5000

static const char* CLASS_LABELS[] = { "guava", "powerbank" };

// ---------------------------------------------------------------------------
// AI Thinker / OV3660 pin map
// ---------------------------------------------------------------------------

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
// Camera init — YUV422 (only colour format OV3660 supports)
// ---------------------------------------------------------------------------

static esp_err_t camera_init(void)
{
    camera_config_t cfg = {};
    cfg.pin_pwdn = CAM_PIN_PWDN; cfg.pin_reset  = CAM_PIN_RESET;
    cfg.pin_xclk = CAM_PIN_XCLK;
    cfg.pin_sccb_sda = CAM_PIN_SIOD; cfg.pin_sccb_scl = CAM_PIN_SIOC;
    cfg.pin_d7 = CAM_PIN_D7; cfg.pin_d6 = CAM_PIN_D6;
    cfg.pin_d5 = CAM_PIN_D5; cfg.pin_d4 = CAM_PIN_D4;
    cfg.pin_d3 = CAM_PIN_D3; cfg.pin_d2 = CAM_PIN_D2;
    cfg.pin_d1 = CAM_PIN_D1; cfg.pin_d0 = CAM_PIN_D0;
    cfg.pin_vsync = CAM_PIN_VSYNC;
    cfg.pin_href  = CAM_PIN_HREF;
    cfg.pin_pclk  = CAM_PIN_PCLK;

    cfg.xclk_freq_hz = 20000000;
    cfg.ledc_timer   = LEDC_TIMER_0;
    cfg.ledc_channel = LEDC_CHANNEL_0;

    cfg.pixel_format = PIXFORMAT_YUV422;
    cfg.frame_size   = FRAMESIZE_96X96;
    cfg.jpeg_quality = 12;
    cfg.fb_count     = 1;
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

    ESP_LOGI(TAG, "Camera init: YUV422 96x96 PSRAM xclk=20MHz");
    esp_err_t ret = esp_camera_init(&cfg);
    if (ret != ESP_OK)
        ESP_LOGE(TAG, "esp_camera_init failed: %s", esp_err_to_name(ret));
    return ret;
}

// ---------------------------------------------------------------------------
// fill_input_tensor_int8
//
// Converts a YUV422 (YUYV) frame buffer into an int8 input tensor.
//
// Step 1 — YUV -> RGB (BT.601 full-range, same as PIL Image.convert("RGB")):
//   R = Y + 1.402   * (V - 128)
//   G = Y - 0.34414 * (U - 128) - 0.71414 * (V - 128)
//   B = Y + 1.772   * (U - 128)
//   Clamp to [0, 255].
//
// Step 2 — uint8 -> int8 (applying the quantization zero_point):
//   int8_val = (int8_t)(uint8_rgb - 128)
//   This is identical to: int8_val = uint8_rgb + zero_point
//   where zero_point = -128 (standard for uint8->int8 with Rescaling(1/255)).
//
//   The division by 255 (Rescaling) is NOT applied here — it is encoded
//   into the quantized weights of the first conv layer. Applying it again
//   would produce values in [-0.5, 0.5] range instead of [-128, 127],
//   which would look like all-black images to the model.
//
// At runtime, always verify the input tensor's actual quantization params:
//   ESP_LOGI(TAG, "input scale=%.6f  zero_point=%d",
//            input->params.scale, (int)(input->params.zero_point));
// Expected: scale≈0.003922 (1/255), zero_point=-128.
// If those values differ, adjust the formula accordingly.
// ---------------------------------------------------------------------------

static void fill_input_tensor_int8(const camera_fb_t* fb, TfLiteTensor* input)
{
    const uint8_t* src = fb->buf;          /* YUYV, 2 bytes per pixel */
    int8_t*        dst = input->data.int8; /* int8 tensor */
    const int      npix = INPUT_W * INPUT_H; /* 9216 pixels, processed 2 at a time */

    for (int i = 0; i < npix; i += 2) {
        /* Unpack one YUYV block (4 bytes = 2 pixels) */
        int Y0 = src[0];
        int U  = src[1];
        int Y1 = src[2];
        int V  = src[3];
        src += 4;

        int u128 = U - 128;
        int v128 = V - 128;

        /* BT.601 YUV->RGB with integer approximation */
        int r0 = clamp255(Y0 + ((179 * v128) >> 7));
        int g0 = clamp255(Y0 - ((44  * u128 + 91 * v128) >> 7));
        int b0 = clamp255(Y0 + ((227 * u128) >> 7));

        int r1 = clamp255(Y1 + ((179 * v128) >> 7));
        int g1 = clamp255(Y1 - ((44  * u128 + 91 * v128) >> 7));
        int b1 = clamp255(Y1 + ((227 * u128) >> 7));

        /* uint8 [0,255] -> int8 [-128,127] by subtracting 128 */
        *dst++ = (int8_t)(r0 - 128);
        *dst++ = (int8_t)(g0 - 128);
        *dst++ = (int8_t)(b0 - 128);
        *dst++ = (int8_t)(r1 - 128);
        *dst++ = (int8_t)(g1 - 128);
        *dst++ = (int8_t)(b1 - 128);
    }
}

// ---------------------------------------------------------------------------
// app_main
// ---------------------------------------------------------------------------

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "EdgeCV TinyML — OV3660 / INT8 Inference");
    ESP_LOGI(TAG, "  Model   : embedded (%u bytes)", model_tflite_len);
    ESP_LOGI(TAG, "  Input   : int8[1,%d,%d,%d]", INPUT_H, INPUT_W, INPUT_C);
    ESP_LOGI(TAG, "  Classes : [0]=%s  [1]=%s",
             CLASS_LABELS[0], CLASS_LABELS[1]);
    ESP_LOGI(TAG, "========================================");
    print_mem("startup");

    // -----------------------------------------------------------------------
    // Parse model
    // -----------------------------------------------------------------------
    const tflite::Model* model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "ABORT: schema version mismatch %lu vs %d",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // -----------------------------------------------------------------------
    // Op resolver
    // Ops used by the scratch TinyML CNN (verified from model inspection):
    //   Conv2D, MaxPool2D, Reshape, FullyConnected, Logistic
    //   Shape + StridedSlice + Pack are generated by Keras's Rescaling layer
    //   during quantization — include them even though they're trivial.
    // -----------------------------------------------------------------------
    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddLogistic();

    // -----------------------------------------------------------------------
    // Tensor arena — try PSRAM first, fall back to internal DRAM
    // The TinyML CNN is much smaller than MobileNetV2; 256 KB should suffice.
    // If AllocateTensors() fails, increase TENSOR_ARENA_SIZE.
    // -----------------------------------------------------------------------
    uint8_t* tensor_arena = (uint8_t*)heap_caps_malloc(
        TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "PSRAM malloc failed, trying internal DRAM");
        tensor_arena = (uint8_t*)heap_caps_malloc(
            TENSOR_ARENA_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (!tensor_arena) {
        ESP_LOGE(TAG, "ABORT: cannot allocate %u KB tensor arena",
                 TENSOR_ARENA_SIZE / 1024);
        return;
    }

    // -----------------------------------------------------------------------
    // Interpreter + AllocateTensors
    // -----------------------------------------------------------------------
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "ABORT: AllocateTensors() failed — "
                      "increase TENSOR_ARENA_SIZE");
        heap_caps_free(tensor_arena);
        return;
    }
    ESP_LOGI(TAG, "AllocateTensors OK — arena used %zu KB of %u KB",
             interpreter.arena_used_bytes() / 1024, TENSOR_ARENA_SIZE / 1024);

    // -----------------------------------------------------------------------
    // Validate input tensor
    // -----------------------------------------------------------------------
    TfLiteTensor* input = interpreter.input(0);

    ESP_LOGI(TAG, "Input  tensor: dims=[%d,%d,%d,%d]  type=%d  "
                  "scale=%.6f  zero_point=%d",
             (int)(input->dims->data[0]), (int)(input->dims->data[1]),
             (int)(input->dims->data[2]), (int)(input->dims->data[3]),
             (int)(input->type),
             input->params.scale, (int)(input->params.zero_point));

    if (input->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "ABORT: expected int8 input tensor (type 9), got type %d",
                 (int)(input->type));
        ESP_LOGE(TAG, "  If model is float32, use fill_input_tensor_float32()");
        heap_caps_free(tensor_arena);
        return;
    }
    if ((int)(input->dims->data[1]) != INPUT_H ||
        (int)(input->dims->data[2]) != INPUT_W ||
        (int)(input->dims->data[3]) != INPUT_C) {
        ESP_LOGE(TAG, "ABORT: shape mismatch — expected [1,%d,%d,%d]",
                 INPUT_H, INPUT_W, INPUT_C);
        heap_caps_free(tensor_arena);
        return;
    }

    /* Log quantization params so they can be verified against expectations */
    ESP_LOGI(TAG, "  Expected: scale≈0.003922  zero_point=-128");
    if ((int)(input->params.zero_point) != -128) {
        ESP_LOGW(TAG, "  WARNING: zero_point is %d, not -128. "
                      "fill_input_tensor_int8() assumes -128. "
                      "Adjust the subtraction if accuracy is poor.",
                 (int)(input->params.zero_point));
    }

    TfLiteTensor* output = interpreter.output(0);
    ESP_LOGI(TAG, "Output tensor: dims=[%d,%d]  type=%d  "
                  "scale=%.6f  zero_point=%d",
             (int)(output->dims->data[0]), (int)(output->dims->data[1]),
             (int)(output->type),
             output->params.scale, (int)(output->params.zero_point));

    print_mem("after AllocateTensors");

    // -----------------------------------------------------------------------
    // Camera init
    // -----------------------------------------------------------------------
    if (camera_init() != ESP_OK) {
        ESP_LOGE(TAG, "ABORT: camera init failed");
        heap_caps_free(tensor_arena);
        return;
    }
    print_mem("after camera init");

    ESP_LOGI(TAG, "Discarding 3 warm-up frames (AGC/AEC stabilisation)...");
    for (int i = 0; i < 3; i++) {
        camera_fb_t* wb = esp_camera_fb_get();
        if (wb) esp_camera_fb_return(wb);
        vTaskDelay(pdMS_TO_TICKS(150));
    }

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Inference loop — every %d ms", INFERENCE_INTERVAL_MS);
    ESP_LOGI(TAG, "========================================");

    const size_t expected_fb_len = (size_t)(INPUT_W * INPUT_H * 2); /* YUV422 */
    uint32_t frame_count = 0;

    while (true) {
        int64_t t_start = esp_timer_get_time();

        // Capture frame
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "[%lu] fb_get returned NULL",
                     (unsigned long)frame_count);
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
            continue;
        }
        if (fb->len != expected_fb_len) {
            ESP_LOGW(TAG, "[%lu] fb->len=%zu expected=%zu — skipping",
                     (unsigned long)frame_count, fb->len, expected_fb_len);
            esp_camera_fb_return(fb);
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
            continue;
        }

        // Preprocess: YUYV -> int8 RGB
        int64_t t_pre = esp_timer_get_time();
        fill_input_tensor_int8(fb, input);
        float pre_ms = (esp_timer_get_time() - t_pre) / 1000.0f;

        esp_camera_fb_return(fb); /* release DMA buffer BEFORE Invoke() */

        // Invoke
        int64_t t_inv = esp_timer_get_time();
        if (interpreter.Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "[%lu] Invoke() FAILED", (unsigned long)frame_count);
            frame_count++;
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
            continue;
        }
        float invoke_ms = (esp_timer_get_time() - t_inv) / 1000.0f;

        // Dequantize output
        // output is int8[1,1] — single sigmoid value
        float powerbank_prob;
        if (output->type == kTfLiteInt8) {
            /* Dequantize: float = (int8_val - zero_point) * scale */
            powerbank_prob =
                (float)(output->data.int8[0] - (int)(output->params.zero_point))
                * output->params.scale;
        } else if (output->type == kTfLiteFloat32) {
            powerbank_prob = output->data.f[0];
        } else {
            ESP_LOGE(TAG, "Unsupported output type: %d", (int)(output->type));
            frame_count++;
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
            continue;
        }

        /* Clamp — dequantization can produce tiny out-of-range values */
        if (powerbank_prob < 0.0f) powerbank_prob = 0.0f;
        if (powerbank_prob > 1.0f) powerbank_prob = 1.0f;

        float guava_prob = 1.0f - powerbank_prob;

        const char* prediction   = (powerbank_prob >= 0.5f) ? "powerbank" : "guava";
        float       confidence   = (powerbank_prob >= 0.5f) ? powerbank_prob : guava_prob;
        float       total_ms     = (esp_timer_get_time() - t_start) / 1000.0f;

        ESP_LOGI(TAG, "--- Frame %lu ---", (unsigned long)frame_count);
        ESP_LOGI(TAG, "  guava     : %.4f  (%.1f%%)", guava_prob,     guava_prob     * 100.0f);
        ESP_LOGI(TAG, "  powerbank : %.4f  (%.1f%%)", powerbank_prob, powerbank_prob * 100.0f);
        ESP_LOGI(TAG, "  PREDICTION : %s  (%.1f%%)", prediction, confidence * 100.0f);
        ESP_LOGI(TAG, "  TIMING     : pre=%.1fms  invoke=%.1fms  total=%.1fms",
                 pre_ms, invoke_ms, total_ms);

        frame_count++;

        int64_t elapsed_ms = (esp_timer_get_time() - t_start) / 1000;
        int64_t sleep_ms   = INFERENCE_INTERVAL_MS - elapsed_ms;
        if (sleep_ms > 0) vTaskDelay(pdMS_TO_TICKS(sleep_ms));
    }
}
