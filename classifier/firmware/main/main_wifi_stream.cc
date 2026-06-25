/*
 * main_wifi_stream.cc
 * ===================
 * EdgeCV — WiFi + MJPEG stream + TinyML inference
 * ESP32-CAM (AI Thinker / OV3660) / ESP-IDF v5.3
 *
 * WHAT THIS DOES:
 *   Streams a live MJPEG feed to a browser and runs TinyML classification
 *   on the same frames — no format switching, no SD card, no stalling.
 *
 * ARCHITECTURE:
 *   One camera config: PIXFORMAT_JPEG, FRAMESIZE_96X96, fb_count=2
 *
 *   Core 1 — camera_task (sole camera owner):
 *     loop:
 *       grab JPEG frame
 *       copy into stream_buf  (mutex)       <- HTTP streamer reads this
 *       every INFER_EVERY_N_FRAMES:
 *         decode JPEG -> RGB565-> int8 tensor -> Invoke()
 *         write result into result_buf (mutex)
 *       return frame to DMA
 *
 *   Core 0 — esp_http_server:
 *     GET /        -> HTML page: <img src=/stream> + JS polling /status
 *     GET /stream  -> MJPEG multipart stream (reads stream_buf)
 *     GET /status  -> JSON {class, confidence, invoke_ms}
 *
 * WHY ONE FORMAT (JPEG) FOR BOTH STREAMING AND INFERENCE:
 *   The OV3660 driver only holds one pixel_format config at a time.
 *   Switching between JPEG (for streaming) and YUV422 (for inference)
 *   requires deinit+reinit (~100ms), which stalls the stream visibly on
 *   every inference cycle. Instead, JPEG frames are decoded to RGB565
 *   on-device using esp_jpg_decode (from espressif/esp_jpeg, already a
 *   transitive dependency of esp32-camera). Decode adds ~20-30ms per
 *   inference frame — negligible against the 184ms Invoke() time.
 *
 * WHY fb_count=2:
 *   Double-buffering lets the DMA fill buffer B while the task reads
 *   buffer A. Without it, esp_camera_fb_get() blocks until the previous
 *   buffer is returned, which would stall the stream during the JPEG
 *   decode + Invoke() cycle.
 *
 * INT8 INPUT PIPELINE (same as main_tinyml_probe.cc):
 *   JPEG decode produces uint8 RGB565.
 *   int8_val = (int8_t)(uint8_pixel - 128)
 *   Do NOT divide by 255 — Rescaling(1/255) is baked into the weights.
 *
 * WIFI CREDENTIALS — edit the two defines below before building.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/event_groups.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_http_server.h"
#include "nvs_flash.h"

#include "esp_camera.h"
#include "img_converters.h"   /* jpg2rgb565 — from esp32-camera component */

#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------------------------------------------------------------------------
// WiFi credentials — EDIT THESE
// ---------------------------------------------------------------------------

#define WIFI_SSID     "idk"
#define WIFI_PASSWORD "lol12345"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

static const char* TAG = "EDGECV";

#define INPUT_W              96
#define INPUT_H              96
#define INPUT_C              3

#define TENSOR_ARENA_SIZE    (256 * 1024)   /* 256 KB — sufficient for TinyML CNN */

/*
 * Run inference every N frames captured.
 * At ~10fps JPEG capture, N=5 -> ~2 inferences per second.
 * Increase N to reduce CPU load on Core 1 if streaming becomes choppy.
 */
#define INFER_EVERY_N_FRAMES  5

#define WIFI_CONNECT_TIMEOUT_MS  15000

static const char* CLASS_LABELS[] = { "guava", "powerbank" };

/* AI Thinker / OV3660 pin map */
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
// Shared state — camera_task writes, HTTP handlers read
// ---------------------------------------------------------------------------

/* Stream buffer — latest JPEG frame for the MJPEG server */
static SemaphoreHandle_t s_stream_mutex  = nullptr;
static uint8_t*          s_stream_buf    = nullptr;
static size_t            s_stream_len    = 0;
static size_t            s_stream_cap    = 0;
static uint32_t          s_stream_seq    = 0;   /* increments each new frame */

/* Inference result */
static SemaphoreHandle_t s_result_mutex  = nullptr;
static char              s_result_class[16] = "initialising";
static float             s_result_conf   = 0.0f;
static float             s_result_inv_ms = 0.0f;
static uint32_t          s_infer_count   = 0;

/* TFLite objects — initialised in app_main, used by camera_task */
static tflite::MicroInterpreter* s_interpreter  = nullptr;
static TfLiteTensor*             s_input_tensor = nullptr;

/* WiFi */
static EventGroupHandle_t s_wifi_eg = nullptr;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1
static int s_wifi_retries = 0;

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
// WiFi
// ---------------------------------------------------------------------------

static void wifi_event_handler(void* arg, esp_event_base_t base,
                               int32_t id, void* data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_wifi_retries < 5) {
            esp_wifi_connect();
            s_wifi_retries++;
            ESP_LOGW(TAG, "WiFi retry %d/5", s_wifi_retries);
        } else {
            xEventGroupSetBits(s_wifi_eg, WIFI_FAIL_BIT);
        }
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* e = (ip_event_got_ip_t*)data;
        ESP_LOGI(TAG, "IP: " IPSTR, IP2STR(&e->ip_info.ip));
        s_wifi_retries = 0;
        xEventGroupSetBits(s_wifi_eg, WIFI_CONNECTED_BIT);
    }
}

static esp_err_t wifi_connect(void)
{
    s_wifi_eg = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t h1, h2;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, nullptr, &h1));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, nullptr, &h2));

    wifi_config_t wc = {};
    strncpy((char*)wc.sta.ssid,     WIFI_SSID,     sizeof(wc.sta.ssid)     - 1);
    strncpy((char*)wc.sta.password, WIFI_PASSWORD, sizeof(wc.sta.password) - 1);
    wc.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to '%s'...", WIFI_SSID);
    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_eg, WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE, pdFALSE, pdMS_TO_TICKS(WIFI_CONNECT_TIMEOUT_MS));

    if (bits & WIFI_CONNECTED_BIT) { ESP_LOGI(TAG, "WiFi connected"); return ESP_OK; }
    ESP_LOGE(TAG, "WiFi connect failed");
    return ESP_FAIL;
}

// ---------------------------------------------------------------------------
// Camera
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

    cfg.pixel_format = PIXFORMAT_JPEG;
    cfg.frame_size   = FRAMESIZE_96X96;
    cfg.jpeg_quality = 12;          /* 0=best, 63=worst; 12 balances size vs quality */
    cfg.fb_count     = 2;           /* double-buffer: DMA fills B while task reads A */
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode    = CAMERA_GRAB_WHEN_EMPTY; /* only grab when a buffer slot is free — prevents FB-OVF */

    ESP_LOGI(TAG, "Camera init: JPEG 96x96 fb_count=2 PSRAM xclk=20MHz");
    esp_err_t ret = esp_camera_init(&cfg);
    if (ret != ESP_OK)
        ESP_LOGE(TAG, "esp_camera_init failed: %s", esp_err_to_name(ret));
    return ret;
}

// ---------------------------------------------------------------------------
// JPEG -> int8 tensor
//
// jpg2rgb565() from the esp32-camera component decodes the JPEG into a
// flat uint16 RGB565 buffer (2 bytes per pixel, packed: RRRRRGGGGGGBBBBB).
// Each pixel is unpacked into 8-bit R, G, B channels (left-shifted to fill
// MSBs), then converted to int8 by subtracting 128 — the correct mapping
// for zero_point=-128.
//
// Buffer size is W*H*2 (RGB565), not W*H*3 (RGB888).
// The RGB565 decode buffer is allocated once in PSRAM and reused across
// inference calls to avoid repeated heap churn.
// ---------------------------------------------------------------------------

static uint8_t* s_rgb_buf = nullptr;  /* 96*96*2 = 18432 bytes RGB565, PSRAM */

static bool jpeg_to_int8_tensor(const uint8_t* jpeg_data, size_t jpeg_len,
                                 TfLiteTensor* input)
{
    const int px = INPUT_W * INPUT_H;

    if (!s_rgb_buf) {
        /* RGB565 = 2 bytes per pixel, not 3 */
        s_rgb_buf = (uint8_t*)heap_caps_malloc(
            px * 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!s_rgb_buf) {
            ESP_LOGE(TAG, "RGB decode buffer alloc failed");
            return false;
        }
    }

    /* Decode JPEG -> RGB565 packed 16-bit pixels into s_rgb_buf */
    bool ok = jpg2rgb565(jpeg_data, jpeg_len, s_rgb_buf, JPG_SCALE_NONE);
    if (!ok) {
        ESP_LOGW(TAG, "jpg2rgb565 decode failed");
        return false;
    }

    /*
     * Unpack RGB565 -> R8 G8 B8 -> int8 tensor (zero_point = -128)
     * RGB565 bit layout: [15:11]=R(5-bit)  [10:5]=G(6-bit)  [4:0]=B(5-bit)
     * Scale each channel to 8-bit by left-shifting (fills MSBs, LSBs zeroed).
     */
    const uint16_t* src = (const uint16_t*)s_rgb_buf;
    int8_t*         dst = input->data.int8;
    for (int i = 0; i < px; i++) {
        uint16_t p = src[i];
        uint8_t r = (uint8_t)(((p >> 11) & 0x1F) << 3);
        uint8_t g = (uint8_t)(((p >>  5) & 0x3F) << 2);
        uint8_t b = (uint8_t)(((p      ) & 0x1F) << 3);
        dst[i * 3 + 0] = (int8_t)((int)r - 128);
        dst[i * 3 + 1] = (int8_t)((int)g - 128);
        dst[i * 3 + 2] = (int8_t)((int)b - 128);
    }
    return true;
}

// ---------------------------------------------------------------------------
// camera_task — Core 1
// Owns the camera exclusively. Grabs JPEG frames, pushes to stream buffer,
// and periodically runs inference.
// ---------------------------------------------------------------------------

static void camera_task(void* arg)
{
    ESP_LOGI(TAG, "camera_task started on core %d", xPortGetCoreID());
    print_mem("camera_task start");

    /* Warm up — discard first few frames while AGC/AEC stabilises */
    for (int i = 0; i < 5; i++) {
        camera_fb_t* wb = esp_camera_fb_get();
        if (wb) esp_camera_fb_return(wb);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    ESP_LOGI(TAG, "Camera warm-up done");

    uint32_t frame_n = 0;

    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb || fb->len == 0) {
            if (fb) esp_camera_fb_return(fb);
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        /* --- Update stream buffer ---------------------------------------- */
        xSemaphoreTake(s_stream_mutex, portMAX_DELAY);
        if (s_stream_cap < fb->len) {
            uint8_t* nb = (uint8_t*)heap_caps_realloc(
                s_stream_buf, fb->len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (nb) { s_stream_buf = nb; s_stream_cap = fb->len; }
        }
        if (s_stream_buf && s_stream_cap >= fb->len) {
            memcpy(s_stream_buf, fb->buf, fb->len);
            s_stream_len = fb->len;
            s_stream_seq++;
        }
        xSemaphoreGive(s_stream_mutex);

        /* --- Run inference every N frames --------------------------------- */
        if (frame_n % INFER_EVERY_N_FRAMES == 0) {
            int64_t t_inv = esp_timer_get_time();

            bool decoded = jpeg_to_int8_tensor(fb->buf, fb->len, s_input_tensor);

            if (decoded && s_interpreter->Invoke() == kTfLiteOk) {
                TfLiteTensor* out = s_interpreter->output(0);

                float powerbank_prob = 0.0f;
                if (out->type == kTfLiteInt8) {
                    powerbank_prob =
                        (float)(out->data.int8[0] - (int)(out->params.zero_point))
                        * out->params.scale;
                } else {
                    powerbank_prob = out->data.f[0];
                }
                powerbank_prob = powerbank_prob < 0.0f ? 0.0f :
                                 powerbank_prob > 1.0f ? 1.0f : powerbank_prob;

                float guava_prob   = 1.0f - powerbank_prob;
                bool  is_powerbank = (powerbank_prob >= 0.5f);
                float conf         = is_powerbank ? powerbank_prob : guava_prob;
                float inv_ms       = (esp_timer_get_time() - t_inv) / 1000.0f;

                xSemaphoreTake(s_result_mutex, portMAX_DELAY);
                strncpy(s_result_class,
                        is_powerbank ? "powerbank" : "guava",
                        sizeof(s_result_class) - 1);
                s_result_conf   = conf;
                s_result_inv_ms = inv_ms;
                s_infer_count++;
                xSemaphoreGive(s_result_mutex);

                ESP_LOGI(TAG, "[infer #%lu] %s %.1f%%  (%.1fms)",
                         (unsigned long)s_infer_count,
                         s_result_class, conf * 100.0f, inv_ms);
            }
        }

        esp_camera_fb_return(fb);
        frame_n++;
    }
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

#define MJPEG_BOUNDARY "edgecvframe"

static esp_err_t stream_handler(httpd_req_t* req)
{
    httpd_resp_set_type(req,
        "multipart/x-mixed-replace;boundary=" MJPEG_BOUNDARY);
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    char hdr[128];
    uint32_t last_seq = 0;

    while (true) {
        /* Wait for a new frame */
        xSemaphoreTake(s_stream_mutex, portMAX_DELAY);
        bool fresh = (s_stream_len > 0 && s_stream_seq != last_seq);

        if (!fresh) {
            xSemaphoreGive(s_stream_mutex);
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        /* Copy frame out quickly so we release the mutex */
        size_t len = s_stream_len;
        uint8_t* tmp = (uint8_t*)malloc(len);   /* internal DRAM — small + fast */
        if (tmp) memcpy(tmp, s_stream_buf, len);
        last_seq = s_stream_seq;
        xSemaphoreGive(s_stream_mutex);

        if (!tmp) { vTaskDelay(pdMS_TO_TICKS(50)); continue; }

        int hlen = snprintf(hdr, sizeof(hdr),
            "--" MJPEG_BOUNDARY "\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: %zu\r\n\r\n", len);

        esp_err_t r = httpd_resp_send_chunk(req, hdr, hlen);
        if (r == ESP_OK) r = httpd_resp_send_chunk(req, (char*)tmp, len);
        if (r == ESP_OK) r = httpd_resp_send_chunk(req, "\r\n", 2);
        free(tmp);

        if (r != ESP_OK) break;   /* client disconnected */
    }
    return ESP_OK;
}

static esp_err_t status_handler(httpd_req_t* req)
{
    char json[128];
    xSemaphoreTake(s_result_mutex, portMAX_DELAY);
    int n = snprintf(json, sizeof(json),
        "{\"class\":\"%s\",\"confidence\":%.3f,\"invoke_ms\":%.1f,\"count\":%lu}",
        s_result_class, s_result_conf, s_result_inv_ms,
        (unsigned long)s_infer_count);
    xSemaphoreGive(s_result_mutex);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, n);
}

static esp_err_t index_handler(httpd_req_t* req)
{
    /*
     * Minimal single-page UI:
     * - Left: MJPEG stream (scaled up 4x for visibility at 96x96 native)
     * - Right: classification label + confidence bar, auto-refreshed every 500ms
     * No external dependencies — pure HTML/CSS/JS served from flash.
     */
    static const char* HTML =
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>EdgeCV</title>"
        "<style>"
        "body{margin:0;background:#111;color:#eee;font-family:monospace;"
        "display:flex;align-items:center;justify-content:center;"
        "height:100vh;gap:32px}"
        "img{width:384px;height:384px;image-rendering:pixelated;"
        "border:2px solid #333;border-radius:4px}"
        ".panel{display:flex;flex-direction:column;gap:16px;min-width:220px}"
        ".label{font-size:2em;font-weight:bold;color:#7cf}"
        ".conf{font-size:1.1em;color:#aaa}"
        ".bar-bg{background:#222;border-radius:4px;height:18px;overflow:hidden}"
        ".bar{height:100%;background:#4af;border-radius:4px;"
        "transition:width 0.3s ease}"
        ".ms{font-size:0.85em;color:#555;margin-top:8px}"
        ".dot{display:inline-block;width:10px;height:10px;border-radius:50%;"
        "background:#4f4;margin-right:6px;animation:pulse 1s infinite}"
        "@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}"
        "</style></head><body>"
        "<img src='/stream' id='feed'/>"
        "<div class='panel'>"
        "<div style='font-size:0.8em;color:#555'>"
        "<span class='dot'></span>EdgeCV Live</div>"
        "<div class='label' id='cls'>...</div>"
        "<div class='conf' id='conf'>confidence</div>"
        "<div class='bar-bg'><div class='bar' id='bar' style='width:0%'></div></div>"
        "<div class='ms' id='ms'>invoke: --ms</div>"
        "</div>"
        "<script>"
        "function poll(){"
        "fetch('/status').then(r=>r.json()).then(d=>{"
        "document.getElementById('cls').textContent=d.class;"
        "document.getElementById('conf').textContent="
        "(d.confidence*100).toFixed(1)+'%';"
        "document.getElementById('bar').style.width="
        "(d.confidence*100).toFixed(1)+'%';"
        "document.getElementById('ms').textContent="
        "'invoke: '+d.invoke_ms.toFixed(0)+'ms | frame #'+d.count;"
        "}).catch(()=>{});"
        "}"
        "setInterval(poll,500);poll();"
        "</script></body></html>";

    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, HTML, strlen(HTML));
}

static httpd_handle_t start_server(void)
{
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.stack_size        = 8192;
    cfg.max_open_sockets  = 4;

    httpd_handle_t server = nullptr;
    if (httpd_start(&server, &cfg) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start HTTP server");
        return nullptr;
    }

    httpd_uri_t uris[] = {
        { "/",       HTTP_GET, index_handler,  nullptr },
        { "/stream", HTTP_GET, stream_handler, nullptr },
        { "/status", HTTP_GET, status_handler, nullptr },
    };
    for (auto& u : uris) httpd_register_uri_handler(server, &u);

    ESP_LOGI(TAG, "HTTP server started — routes: /  /stream  /status");
    return server;
}

// ---------------------------------------------------------------------------
// app_main
// ---------------------------------------------------------------------------

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "EdgeCV — WiFi + MJPEG + TinyML Inference");
    ESP_LOGI(TAG, "  SSID   : %s", WIFI_SSID);
    ESP_LOGI(TAG, "  Model  : embedded (%u bytes)", model_tflite_len);
    ESP_LOGI(TAG, "  Infer  : every %d frames", INFER_EVERY_N_FRAMES);
    ESP_LOGI(TAG, "========================================");
    print_mem("startup");

    /* NVS — required by WiFi driver */
    ESP_ERROR_CHECK(nvs_flash_init());

    /* Mutexes */
    s_stream_mutex = xSemaphoreCreateMutex();
    s_result_mutex = xSemaphoreCreateMutex();

    /* ---- TFLite setup ---------------------------------------------------- */
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
    if (!arena) {
        ESP_LOGE(TAG, "ABORT: tensor arena malloc failed"); return;
    }

    static tflite::MicroInterpreter interpreter(model, resolver, arena, TENSOR_ARENA_SIZE);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "ABORT: AllocateTensors failed"); return;
    }
    ESP_LOGI(TAG, "AllocateTensors OK — arena used %zu KB",
             interpreter.arena_used_bytes() / 1024);

    s_interpreter  = &interpreter;
    s_input_tensor = interpreter.input(0);

    /* Validate input tensor */
    if (s_input_tensor->type != kTfLiteInt8 ||
        (int)(s_input_tensor->dims->data[1]) != INPUT_H ||
        (int)(s_input_tensor->dims->data[2]) != INPUT_W ||
        (int)(s_input_tensor->dims->data[3]) != INPUT_C) {
        ESP_LOGE(TAG, "ABORT: input tensor mismatch");
        return;
    }
    ESP_LOGI(TAG, "Input tensor OK — int8[1,%d,%d,%d]  zero_point=%d",
             INPUT_H, INPUT_W, INPUT_C,
             (int)(s_input_tensor->params.zero_point));
    print_mem("after TFLite init");

    /* ---- Camera ---------------------------------------------------------- */
    if (camera_init() != ESP_OK) {
        ESP_LOGE(TAG, "ABORT: camera init failed"); return;
    }
    print_mem("after camera init");

    /* ---- WiFi ------------------------------------------------------------ */
    if (wifi_connect() != ESP_OK) {
        ESP_LOGE(TAG, "ABORT: WiFi connect failed — check SSID/password"); return;
    }
    print_mem("after WiFi connect");

    /* ---- HTTP server ----------------------------------------------------- */
    if (start_server() == nullptr) {
        ESP_LOGE(TAG, "ABORT: HTTP server failed"); return;
    }

    /* ---- Camera task on Core 1 ------------------------------------------ */
    xTaskCreatePinnedToCore(
        camera_task, "camera_task",
        8192,           /* stack — needs room for JPEG decode */
        nullptr, 5,     /* priority 5 */
        nullptr, 1);    /* Core 1 */

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Ready. Open http://<device-ip>/ in a browser.");
    ESP_LOGI(TAG, "  /         -> live view + classification");
    ESP_LOGI(TAG, "  /stream   -> raw MJPEG");
    ESP_LOGI(TAG, "  /status   -> JSON classification result");
    ESP_LOGI(TAG, "========================================");

    /* Main task — periodic memory report */
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(30000));
        print_mem("periodic");
    }
}
