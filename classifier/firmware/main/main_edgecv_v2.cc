/*
 * main_edgecv_v2.cc
 * =================
 * EdgeCV Phase 2 — Performance + Open-World Rejection + Temporal Stability
 * ESP32-CAM (AI Thinker / OV3660) / ESP-IDF v5.3
 *
 * ============================================================
 * PHASE A — PERFORMANCE
 * ============================================================
 * Every pipeline stage is timed independently and printed every
 * PROFILE_EVERY_N_FRAMES frames so you can measure actual breakdowns
 * on your hardware rather than relying on estimates.
 *
 * Input resolution is configurable via INPUT_W / INPUT_H.
 * To move to 64x64: change both to 64, retrain on 64x64, re-embed model.
 * Expected Invoke() improvement: ~135ms -> ~65ms (pixel count halves twice).
 *
 * PHASE B — OPEN-WORLD REJECTION
 * ============================================================
 * Three rejection strategies implemented and selectable at compile time:
 *
 *   REJECT_MODE_THRESHOLD (default):
 *     If max(guava_prob, powerbank_prob) < CONFIDENCE_THRESHOLD -> UNKNOWN
 *     Simple. Works well when known-class confidence is reliably high (>0.8).
 *     Recommended starting point.
 *
 *   REJECT_MODE_MARGIN:
 *     If |guava_prob - powerbank_prob| < MARGIN_THRESHOLD -> UNKNOWN
 *     Catches cases where the model is split but both scores are moderate.
 *     Use when threshold alone passes too many unknowns at ~0.6 confidence.
 *
 *   REJECT_MODE_COMBINED:
 *     Both threshold AND margin must be satisfied.
 *     Most conservative — use when false positives on unknowns are costly.
 *
 * Entropy-based rejection is NOT implemented: for a 2-class sigmoid model,
 * entropy = -p*log(p) - (1-p)*log(1-p), which is a monotonic function of
 * |p - 0.5|. It is mathematically equivalent to the margin approach and
 * adds no new information with 2 classes. Worth revisiting for 3+ classes.
 *
 * PHASE C — TEMPORAL STABILIZATION
 * ============================================================
 * Four strategies implemented, selectable via STABILITY_MODE:
 *
 *   STABILITY_MAJORITY_VOTE:
 *     Keeps a ring buffer of N raw predictions. Output = most common label.
 *     Fast to respond to genuine changes. Sensitive to single-frame noise.
 *
 *   STABILITY_CONF_AVG (recommended):
 *     Averages the raw probability scores over N frames. Threshold applied
 *     to the average. Smoother than vote, still responds quickly.
 *     Best for most deployment scenarios.
 *
 *   STABILITY_EMA:
 *     Exponential moving average of probability score. ALPHA controls
 *     weight of new frame vs history. Lower ALPHA = more smoothing.
 *     Good for continuous streams where abrupt changes are rare.
 *
 *   STABILITY_HYSTERESIS:
 *     Once a label is confirmed, requires the confidence to drop below
 *     HYSTERESIS_LOW before switching. Prevents rapid alternation at
 *     decision boundaries. Best for motor/actuator control where flapping
 *     is physically harmful.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#include "esp_camera.h"
#include "img_converters.h"

#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* TAG = "EDGECV2";

// ===========================================================================
// CONFIGURATION — edit these before building
// ===========================================================================

/* Input resolution — must match the model embedded in model_data.cc.
 * Current default: 96x96 (existing model).
 * For 64x64: change both, retrain with IMG_HEIGHT=IMG_WIDTH=64,
 * re-embed with xxd, rebuild. Expected speedup: ~2x on Invoke(). */
#define INPUT_W   96
#define INPUT_H   96
#define INPUT_C   3

#define TENSOR_ARENA_SIZE   (256 * 1024)
#define INFERENCE_INTERVAL_MS  200   /* ms between inference attempts */
#define PROFILE_EVERY_N_FRAMES  10   /* print timing breakdown every N frames */

/* --------------------------------------------------------------------------
 * PHASE B — rejection mode (pick one)
 * -------------------------------------------------------------------------- */
#define REJECT_MODE_THRESHOLD   0   /* confidence of winning class < threshold */
#define REJECT_MODE_MARGIN      1   /* |class0 - class1| < margin */
#define REJECT_MODE_COMBINED    2   /* both threshold AND margin */

#define REJECTION_MODE   REJECT_MODE_COMBINED   /* <-- change this */

/* Threshold: winning class must exceed this to be accepted.
 * Start at 0.75. Raise if unknowns (wall, ceiling) get classified.
 * Lower if known objects are incorrectly rejected. */
#define CONFIDENCE_THRESHOLD  0.75f

/* Margin: difference between class scores must exceed this.
 * A wall might give guava=0.55, powerbank=0.45 -> margin=0.10 -> UNKNOWN.
 * Start at 0.40. */
#define MARGIN_THRESHOLD  0.40f

/* --------------------------------------------------------------------------
 * PHASE C — stabilization mode (pick one)
 * -------------------------------------------------------------------------- */
#define STABILITY_MAJORITY_VOTE  0
#define STABILITY_CONF_AVG       1   /* recommended */
#define STABILITY_EMA            2
#define STABILITY_HYSTERESIS     3

#define STABILITY_MODE   STABILITY_CONF_AVG   /* <-- change this */

/* Window size for vote / conf_avg. Larger = smoother, slower to respond. */
#define STABILITY_WINDOW  5

/* EMA alpha: weight of the new frame. 0.0=ignore new, 1.0=no smoothing.
 * 0.3 means new frame contributes 30%, history 70%. */
#define EMA_ALPHA  0.30f

/* Hysteresis: once confirmed, confidence must drop below LOW to switch. */
#define HYSTERESIS_HIGH  0.70f   /* must exceed this to confirm a class */
#define HYSTERESIS_LOW   0.45f   /* must drop below this to release it */

/* Class labels — must match training folder alphabetical order */
static const char* CLASS_LABELS[] = { "guava", "powerbank" };
#define CLASS_UNKNOWN  "UNKNOWN"

/* AI Thinker / OV3660 pins */
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

// ===========================================================================
// Helpers
// ===========================================================================

static void print_mem(const char* label)
{
    ESP_LOGI(TAG, "MEM [%s]  heap=%zu KB  psram=%zu KB", label,
             heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024,
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM)   / 1024);
}

// ===========================================================================
// PHASE A — Profiler
// Accumulates per-stage timing across frames and reports averages.
// ===========================================================================

typedef struct {
    float capture_ms;
    float decode_ms;
    float convert_ms;
    float invoke_ms;
    float postproc_ms;
    float total_ms;
} FrameTiming;

static FrameTiming s_timing_acc = {};
static uint32_t    s_timing_n   = 0;

static void timing_accumulate(const FrameTiming* t)
{
    s_timing_acc.capture_ms  += t->capture_ms;
    s_timing_acc.decode_ms   += t->decode_ms;
    s_timing_acc.convert_ms  += t->convert_ms;
    s_timing_acc.invoke_ms   += t->invoke_ms;
    s_timing_acc.postproc_ms += t->postproc_ms;
    s_timing_acc.total_ms    += t->total_ms;
    s_timing_n++;
}

static void timing_report_and_reset(void)
{
    if (s_timing_n == 0) return;
    float n = (float)s_timing_n;
    ESP_LOGI(TAG, "=== TIMING PROFILE (avg over %lu frames) ===",
             (unsigned long)s_timing_n);
    ESP_LOGI(TAG, "  capture  : %6.2f ms  (%4.1f%%)",
             s_timing_acc.capture_ms  / n,
             100.f * s_timing_acc.capture_ms  / s_timing_acc.total_ms);
    ESP_LOGI(TAG, "  decode   : %6.2f ms  (%4.1f%%)",
             s_timing_acc.decode_ms   / n,
             100.f * s_timing_acc.decode_ms   / s_timing_acc.total_ms);
    ESP_LOGI(TAG, "  convert  : %6.2f ms  (%4.1f%%)",
             s_timing_acc.convert_ms  / n,
             100.f * s_timing_acc.convert_ms  / s_timing_acc.total_ms);
    ESP_LOGI(TAG, "  invoke   : %6.2f ms  (%4.1f%%)",
             s_timing_acc.invoke_ms   / n,
             100.f * s_timing_acc.invoke_ms   / s_timing_acc.total_ms);
    ESP_LOGI(TAG, "  postproc : %6.2f ms  (%4.1f%%)",
             s_timing_acc.postproc_ms / n,
             100.f * s_timing_acc.postproc_ms / s_timing_acc.total_ms);
    ESP_LOGI(TAG, "  TOTAL    : %6.2f ms  -> %.1f FPS",
             s_timing_acc.total_ms / n,
             1000.f / (s_timing_acc.total_ms / n));
    ESP_LOGI(TAG, "=============================================");
    memset(&s_timing_acc, 0, sizeof(s_timing_acc));
    s_timing_n = 0;
}

// ===========================================================================
// PHASE B — Open-World Rejection
// ===========================================================================

typedef enum {
    PRED_GUAVA     = 0,
    PRED_POWERBANK = 1,
    PRED_UNKNOWN   = 2,
} Prediction;

static Prediction apply_rejection(float powerbank_prob)
{
    float guava_prob = 1.0f - powerbank_prob;
    float conf       = (powerbank_prob >= 0.5f) ? powerbank_prob : guava_prob;
    float margin     = fabsf(powerbank_prob - guava_prob);

#if   REJECTION_MODE == REJECT_MODE_THRESHOLD
    if (conf < CONFIDENCE_THRESHOLD) return PRED_UNKNOWN;

#elif REJECTION_MODE == REJECT_MODE_MARGIN
    if (margin < MARGIN_THRESHOLD)   return PRED_UNKNOWN;

#elif REJECTION_MODE == REJECT_MODE_COMBINED
    if (conf   < CONFIDENCE_THRESHOLD) return PRED_UNKNOWN;
    if (margin < MARGIN_THRESHOLD)     return PRED_UNKNOWN;
#endif

    return (powerbank_prob >= 0.5f) ? PRED_POWERBANK : PRED_GUAVA;
}

static const char* prediction_label(Prediction p)
{
    if (p == PRED_UNKNOWN) return CLASS_UNKNOWN;
    return CLASS_LABELS[(int)p];
}

// ===========================================================================
// PHASE C — Temporal Stabilization
// ===========================================================================

/* --- Majority Vote -------------------------------------------------------- */
static Prediction s_vote_buf[STABILITY_WINDOW] = {};
static int        s_vote_idx = 0;
static bool       s_vote_full = false;

static Prediction stabilize_majority_vote(Prediction raw)
{
    s_vote_buf[s_vote_idx] = raw;
    s_vote_idx = (s_vote_idx + 1) % STABILITY_WINDOW;
    if (s_vote_idx == 0) s_vote_full = true;

    int len = s_vote_full ? STABILITY_WINDOW : s_vote_idx;
    int counts[3] = {};
    for (int i = 0; i < len; i++) counts[(int)s_vote_buf[i]]++;

    int best = 0;
    for (int i = 1; i < 3; i++)
        if (counts[i] > counts[best]) best = i;
    return (Prediction)best;
}

/* --- Confidence Average --------------------------------------------------- */
static float s_prob_buf[STABILITY_WINDOW] = {};
static int   s_prob_idx  = 0;
static bool  s_prob_full = false;

static Prediction stabilize_conf_avg(float powerbank_prob)
{
    s_prob_buf[s_prob_idx] = powerbank_prob;
    s_prob_idx = (s_prob_idx + 1) % STABILITY_WINDOW;
    if (s_prob_idx == 0) s_prob_full = true;

    int   len = s_prob_full ? STABILITY_WINDOW : s_prob_idx;
    float sum = 0.f;
    for (int i = 0; i < len; i++) sum += s_prob_buf[i];
    float avg = sum / (float)len;

    return apply_rejection(avg);
}

/* --- Exponential Moving Average ------------------------------------------ */
static float s_ema = 0.5f;   /* initialised to maximum uncertainty */
static bool  s_ema_init = false;

static Prediction stabilize_ema(float powerbank_prob)
{
    if (!s_ema_init) { s_ema = powerbank_prob; s_ema_init = true; }
    else             { s_ema = EMA_ALPHA * powerbank_prob + (1.f - EMA_ALPHA) * s_ema; }
    return apply_rejection(s_ema);
}

/* --- Hysteresis ----------------------------------------------------------- */
static Prediction s_hyst_state     = PRED_UNKNOWN;
static float      s_hyst_conf_high = 0.f;

static Prediction stabilize_hysteresis(float powerbank_prob)
{
    float guava_prob = 1.0f - powerbank_prob;
    float conf       = (powerbank_prob >= 0.5f) ? powerbank_prob : guava_prob;
    Prediction raw   = apply_rejection(powerbank_prob);

    if (s_hyst_state == PRED_UNKNOWN) {
        /* Only confirm a class when confidence is clearly above HIGH */
        if (raw != PRED_UNKNOWN && conf >= HYSTERESIS_HIGH) {
            s_hyst_state = raw;
        }
    } else {
        /* Hold current state until confidence drops below LOW */
        float state_conf = (s_hyst_state == PRED_POWERBANK)
                           ? powerbank_prob : guava_prob;
        if (state_conf < HYSTERESIS_LOW) {
            s_hyst_state = PRED_UNKNOWN;
        } else if (raw != PRED_UNKNOWN && raw != s_hyst_state && conf >= HYSTERESIS_HIGH) {
            /* Confident flip to other class */
            s_hyst_state = raw;
        }
    }
    return s_hyst_state;
}

/* --- Dispatch ------------------------------------------------------------ */
static Prediction stabilize(float powerbank_prob)
{
#if   STABILITY_MODE == STABILITY_MAJORITY_VOTE
    return stabilize_majority_vote(apply_rejection(powerbank_prob));
#elif STABILITY_MODE == STABILITY_CONF_AVG
    return stabilize_conf_avg(powerbank_prob);
#elif STABILITY_MODE == STABILITY_EMA
    return stabilize_ema(powerbank_prob);
#elif STABILITY_MODE == STABILITY_HYSTERESIS
    return stabilize_hysteresis(powerbank_prob);
#else
    return apply_rejection(powerbank_prob);
#endif
}

// ===========================================================================
// Camera
// ===========================================================================

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

    cfg.pixel_format = PIXFORMAT_JPEG;
    cfg.frame_size   = FRAMESIZE_96X96;
    cfg.jpeg_quality = 12;
    cfg.fb_count     = 1;
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

    ESP_LOGI(TAG, "Camera: JPEG %dx%d fb_count=1 PSRAM xclk=20MHz",
             INPUT_W, INPUT_H);
    esp_err_t ret = esp_camera_init(&cfg);
    if (ret != ESP_OK)
        ESP_LOGE(TAG, "camera_init failed: %s", esp_err_to_name(ret));
    return ret;
}

// ===========================================================================
// JPEG decode + int8 conversion
// ===========================================================================

static uint8_t* s_rgb_buf = nullptr;

static bool jpeg_to_int8(const uint8_t* jpeg, size_t jpeg_len,
                          TfLiteTensor* input)
{
    if (!s_rgb_buf) {
        s_rgb_buf = (uint8_t*)heap_caps_malloc(
            INPUT_W * INPUT_H * INPUT_C, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!s_rgb_buf) { ESP_LOGE(TAG, "rgb_buf alloc fail"); return false; }
    }

    if (!jpg2rgb888(jpeg, jpeg_len, s_rgb_buf, JPG_SCALE_NONE)) {
        ESP_LOGW(TAG, "jpg2rgb888 failed");
        return false;
    }

    const uint8_t* src = s_rgb_buf;
    int8_t*        dst = input->data.int8;
    const int      n   = INPUT_W * INPUT_H * INPUT_C;
    for (int i = 0; i < n; i++)
        dst[i] = (int8_t)((int)src[i] - 128);

    return true;
}

// ===========================================================================
// app_main
// ===========================================================================

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "EdgeCV Phase 2");
    ESP_LOGI(TAG, "  Input     : %dx%d int8", INPUT_W, INPUT_H);
    ESP_LOGI(TAG, "  Rejection : mode=%d  thresh=%.2f  margin=%.2f",
             REJECTION_MODE, CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD);
    ESP_LOGI(TAG, "  Stability : mode=%d  window=%d",
             STABILITY_MODE, STABILITY_WINDOW);
    ESP_LOGI(TAG, "========================================");
    print_mem("startup");

    /* TFLite setup */
    const tflite::Model* model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "ABORT: schema mismatch"); return;
    }

    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddLogistic();

    uint8_t* arena = (uint8_t*)heap_caps_malloc(
        TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!arena) { ESP_LOGE(TAG, "ABORT: arena alloc fail"); return; }

    static tflite::MicroInterpreter interpreter(
        model, resolver, arena, TENSOR_ARENA_SIZE);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "ABORT: AllocateTensors failed"); return;
    }
    ESP_LOGI(TAG, "AllocateTensors OK — arena used %zu KB",
             interpreter.arena_used_bytes() / 1024);

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    if (input->type != kTfLiteInt8 ||
        (int)(input->dims->data[1]) != INPUT_H ||
        (int)(input->dims->data[2]) != INPUT_W) {
        ESP_LOGE(TAG, "ABORT: input tensor mismatch — "
                      "got [%d,%d,%d] expected [%d,%d,%d]",
                 (int)(input->dims->data[1]),
                 (int)(input->dims->data[2]),
                 (int)(input->dims->data[3]),
                 INPUT_H, INPUT_W, INPUT_C);
        return;
    }
    ESP_LOGI(TAG, "Input  OK: int8[1,%d,%d,%d] zp=%d scale=%.6f",
             INPUT_H, INPUT_W, INPUT_C,
             (int)(input->params.zero_point), input->params.scale);
    ESP_LOGI(TAG, "Output OK: type=%d zp=%d scale=%.6f",
             (int)(output->type),
             (int)(output->params.zero_point), output->params.scale);
    print_mem("after TFLite init");

    if (camera_init() != ESP_OK) { ESP_LOGE(TAG, "ABORT: camera"); return; }
    print_mem("after camera init");

    /* Warm up */
    ESP_LOGI(TAG, "Warm-up frames...");
    for (int i = 0; i < 5; i++) {
        camera_fb_t* wb = esp_camera_fb_get();
        if (wb) esp_camera_fb_return(wb);
        vTaskDelay(pdMS_TO_TICKS(100));
    }

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Running. Profile prints every %d frames.",
             PROFILE_EVERY_N_FRAMES);
    ESP_LOGI(TAG, "========================================");

    uint32_t frame_n = 0;

    while (true) {
        FrameTiming t = {};
        int64_t t0, t1;

        /* ---- CAPTURE ---------------------------------------------------- */
        t0 = esp_timer_get_time();
        camera_fb_t* fb = esp_camera_fb_get();
        t1 = esp_timer_get_time();
        t.capture_ms = (t1 - t0) / 1000.f;

        if (!fb || fb->len == 0) {
            if (fb) esp_camera_fb_return(fb);
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        /* ---- DECODE JPEG -> RGB888 --------------------------------------- */
        t0 = esp_timer_get_time();
        bool decoded = (s_rgb_buf || (s_rgb_buf = (uint8_t*)heap_caps_malloc(
            INPUT_W * INPUT_H * INPUT_C, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)));
        if (decoded)
            decoded = jpg2rgb888(fb->buf, fb->len, s_rgb_buf, JPG_SCALE_NONE);
        t1 = esp_timer_get_time();
        t.decode_ms = (t1 - t0) / 1000.f;

        /* ---- UINT8 -> INT8 CONVERSION ------------------------------------ */
        t0 = esp_timer_get_time();
        if (decoded) {
            const uint8_t* src = s_rgb_buf;
            int8_t*        dst = input->data.int8;
            const int      n   = INPUT_W * INPUT_H * INPUT_C;
            for (int i = 0; i < n; i++)
                dst[i] = (int8_t)((int)src[i] - 128);
        }
        t1 = esp_timer_get_time();
        t.convert_ms = (t1 - t0) / 1000.f;

        esp_camera_fb_return(fb);  /* release DMA before Invoke() */

        if (!decoded) {
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
            continue;
        }

        /* ---- INVOKE ----------------------------------------------------- */
        t0 = esp_timer_get_time();
        TfLiteStatus status = interpreter.Invoke();
        t1 = esp_timer_get_time();
        t.invoke_ms = (t1 - t0) / 1000.f;

        if (status != kTfLiteOk) {
            ESP_LOGE(TAG, "[%lu] Invoke() failed", (unsigned long)frame_n);
            frame_n++;
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
            continue;
        }

        /* ---- POST-PROCESSING (Phase B + C) ------------------------------ */
        t0 = esp_timer_get_time();

        /* Dequantize output */
        float powerbank_prob;
        if (output->type == kTfLiteInt8) {
            powerbank_prob =
                (float)(output->data.int8[0] - (int)(output->params.zero_point))
                * output->params.scale;
        } else {
            powerbank_prob = output->data.f[0];
        }
        powerbank_prob = powerbank_prob < 0.f ? 0.f :
                         powerbank_prob > 1.f ? 1.f : powerbank_prob;
        float guava_prob = 1.f - powerbank_prob;

        /* Phase B — rejection (raw, before stabilization) */
        Prediction raw_pred = apply_rejection(powerbank_prob);

        /* Phase C — stabilized output */
        Prediction stable_pred = stabilize(powerbank_prob);

        float conf = (powerbank_prob >= 0.5f) ? powerbank_prob : guava_prob;

        t1 = esp_timer_get_time();
        t.postproc_ms = (t1 - t0) / 1000.f;

        /* Total */
        t.total_ms = t.capture_ms + t.decode_ms + t.convert_ms
                   + t.invoke_ms  + t.postproc_ms;

        /* ---- PRINT RESULT ----------------------------------------------- */
        ESP_LOGI(TAG, "--- Frame %lu ---", (unsigned long)frame_n);
        ESP_LOGI(TAG, "  guava     : %.4f  (%.1f%%)", guava_prob,     guava_prob     * 100.f);
        ESP_LOGI(TAG, "  powerbank : %.4f  (%.1f%%)", powerbank_prob, powerbank_prob * 100.f);
        ESP_LOGI(TAG, "  raw       : %s", prediction_label(raw_pred));
        ESP_LOGI(TAG, "  stable    : %s  <- use this", prediction_label(stable_pred));
        ESP_LOGI(TAG, "  invoke    : %.1f ms", t.invoke_ms);

        /* ---- PROFILE REPORT --------------------------------------------- */
        timing_accumulate(&t);
        if ((frame_n + 1) % PROFILE_EVERY_N_FRAMES == 0) {
            timing_report_and_reset();
        }

        frame_n++;

        int64_t elapsed_ms = (int64_t)(t.total_ms);
        int64_t sleep_ms   = (int64_t)INFERENCE_INTERVAL_MS - elapsed_ms;
        if (sleep_ms > 0) vTaskDelay(pdMS_TO_TICKS((uint32_t)sleep_ms));
    }
}
