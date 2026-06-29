/*
 * main_stream_raw.cc
 * ==================
 * EdgeCV — Live camera stream over TCP + TinyML inference printed to serial.
 * ESP32-CAM (AI Thinker / OV3660) / ESP-IDF v5.3
 *
 * WHAT THIS DOES:
 *   - Captures 96x96 YUV422 frames from the OV3660
 *   - Encodes each frame to JPEG using frame2jpg()
 *   - Sends frames over a raw TCP socket (4-byte LE length prefix + JPEG bytes)
 *   - Runs TinyML inference on every Nth frame and prints predictions to serial
 *   - Advertises itself via mDNS as classifier-espcam.local:5555
 *
 * WHAT THE LAPTOP DOES (live_infer.py):
 *   - Connects to classifier-espcam.local:5555
 *   - Decodes incoming JPEG frames and displays them in an OpenCV window
 *   - No inference on the laptop side — the ESP32 does all classification
 *
 * PROTOCOL:
 *   [uint32_t frame_len LE] [JPEG bytes × frame_len]  (repeated)
 *
 * CAMERA FORMAT:
 *   YUV422 96x96 — same format used by main_tinyml_probe.cc for inference.
 *   frame2jpg() encodes the YUV frame to JPEG for transmission.
 *   For inference, the YUV frame is decoded to RGB before tensor fill
 *   (same BT.601 YUV→RGB conversion used throughout the project).
 *
 * INT8 INPUT:
 *   int8_val = (int8_t)(uint8_rgb_pixel - 128)
 *   Rescaling(1/255) is baked into the model weights — do not divide by 255.
 *
 * WIFI CREDENTIALS — edit the two defines below.
 * MDNS_HOSTNAME — what the device advertises as on your local network.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"

#include "lwip/sockets.h"
#include "lwip/inet.h"

#include "esp_camera.h"
#include "img_converters.h"   /* frame2jpg */

#include "mdns.h"

#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------------------------------------------------------------------------
// Configuration — edit these
// ---------------------------------------------------------------------------

#define WIFI_SSID       "idk"
#define WIFI_PASSWORD   "lol12345"
#define STREAM_PORT     5555
#define MDNS_HOSTNAME   "classifier-espcam"   /* reachable as classifier-espcam.local */

/*
 * Run inference every N frames.
 * At ~5fps stream rate, N=3 gives ~1.5 inferences per second.
 * Increase to reduce serial noise; decrease for faster prediction updates.
 */
#define INFER_EVERY_N_FRAMES  3

#define TENSOR_ARENA_SIZE  (256 * 1024)

static const char* CLASS_LABELS[] = { "guava", "powerbank" };

#define INPUT_W  96
#define INPUT_H  96
#define INPUT_C  3

// ---------------------------------------------------------------------------
// OV3660 pin map (AI Thinker ESP32-CAM)
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

static const char* TAG = "RAW_STREAM";

// ---------------------------------------------------------------------------
// WiFi
// ---------------------------------------------------------------------------

static EventGroupHandle_t s_wifi_eg;
#define WIFI_CONNECTED_BIT BIT0

static void wifi_event_handler(void*, esp_event_base_t base,
                               int32_t id, void* data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* e = (ip_event_got_ip_t*)data;
        ESP_LOGI(TAG, "IP: " IPSTR, IP2STR(&e->ip_info.ip));
        xEventGroupSetBits(s_wifi_eg, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init(void)
{
    s_wifi_eg = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,  wifi_event_handler, NULL);
    esp_event_handler_register(IP_EVENT,   IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL);

    wifi_config_t wc = {};
    strncpy((char*)wc.sta.ssid,     WIFI_SSID,     sizeof(wc.sta.ssid)     - 1);
    strncpy((char*)wc.sta.password, WIFI_PASSWORD, sizeof(wc.sta.password) - 1);
    wc.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_connect();

    xEventGroupWaitBits(s_wifi_eg, WIFI_CONNECTED_BIT,
                        pdFALSE, pdTRUE, portMAX_DELAY);
}

// ---------------------------------------------------------------------------
// mDNS
// ---------------------------------------------------------------------------

static void mdns_init_service(void)
{
    ESP_ERROR_CHECK(mdns_init());
    ESP_ERROR_CHECK(mdns_hostname_set(MDNS_HOSTNAME));
    ESP_ERROR_CHECK(mdns_instance_name_set("EdgeCV Raw Stream"));

    mdns_txt_item_t txt[] = {
        {"version", "1"},
        {"port",    "5555"},
    };
    ESP_ERROR_CHECK(mdns_service_add(
        "EdgeCV-Stream", "_edgecv", "_tcp",
        STREAM_PORT, txt, sizeof(txt) / sizeof(txt[0])
    ));

    ESP_LOGI(TAG, "mDNS: hostname=%s.local  service=_edgecv._tcp port=%d",
             MDNS_HOSTNAME, STREAM_PORT);
}

// ---------------------------------------------------------------------------
// Camera — YUV422 96x96 (same config as tinyml_probe)
// ---------------------------------------------------------------------------

static esp_err_t camera_init(void)
{
    camera_config_t cfg = {};
    cfg.pin_pwdn     = CAM_PIN_PWDN;
    cfg.pin_reset    = CAM_PIN_RESET;
    cfg.pin_xclk     = CAM_PIN_XCLK;
    cfg.pin_sccb_sda = CAM_PIN_SIOD;
    cfg.pin_sccb_scl = CAM_PIN_SIOC;
    cfg.pin_d7 = CAM_PIN_D7; cfg.pin_d6 = CAM_PIN_D6;
    cfg.pin_d5 = CAM_PIN_D5; cfg.pin_d4 = CAM_PIN_D4;
    cfg.pin_d3 = CAM_PIN_D3; cfg.pin_d2 = CAM_PIN_D2;
    cfg.pin_d1 = CAM_PIN_D1; cfg.pin_d0 = CAM_PIN_D0;
    cfg.pin_vsync    = CAM_PIN_VSYNC;
    cfg.pin_href     = CAM_PIN_HREF;
    cfg.pin_pclk     = CAM_PIN_PCLK;

    cfg.xclk_freq_hz = 20000000;
    cfg.ledc_timer   = LEDC_TIMER_0;
    cfg.ledc_channel = LEDC_CHANNEL_0;

    cfg.pixel_format = PIXFORMAT_YUV422;     /* raw YUV — avoids JPEG encode pipeline issues */
    cfg.frame_size   = FRAMESIZE_96X96;
    cfg.jpeg_quality = 12;
    cfg.fb_count     = 2;                    /* double-buffer so DMA doesn't stall */
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode    = CAMERA_GRAB_WHEN_EMPTY; /* backpressure-safe — no FB-OVF */

    ESP_LOGI(TAG, "Camera: YUV422 96x96 fb_count=2 PSRAM");
    esp_err_t ret = esp_camera_init(&cfg);
    if (ret != ESP_OK)
        ESP_LOGE(TAG, "camera_init failed: %s", esp_err_to_name(ret));
    return ret;
}

// ---------------------------------------------------------------------------
// YUV422 → int8 tensor
//
// Same BT.601 conversion used in main_tinyml_probe.cc.
// Processes 2 pixels per loop iteration (YUYV block).
// ---------------------------------------------------------------------------

static inline int clamp255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

static void yuv422_to_int8_tensor(const camera_fb_t* fb, TfLiteTensor* input)
{
    const uint8_t* src = fb->buf;
    int8_t*        dst = input->data.int8;
    const int      npix = INPUT_W * INPUT_H;

    for (int i = 0; i < npix; i += 2) {
        int Y0 = src[0], U = src[1], Y1 = src[2], V = src[3];
        src += 4;

        int u128 = U - 128, v128 = V - 128;

        int r0 = clamp255(Y0 + ((179 * v128) >> 7));
        int g0 = clamp255(Y0 - ((44  * u128 + 91 * v128) >> 7));
        int b0 = clamp255(Y0 + ((227 * u128) >> 7));
        int r1 = clamp255(Y1 + ((179 * v128) >> 7));
        int g1 = clamp255(Y1 - ((44  * u128 + 91 * v128) >> 7));
        int b1 = clamp255(Y1 + ((227 * u128) >> 7));

        *dst++ = (int8_t)(r0 - 128); *dst++ = (int8_t)(g0 - 128); *dst++ = (int8_t)(b0 - 128);
        *dst++ = (int8_t)(r1 - 128); *dst++ = (int8_t)(g1 - 128); *dst++ = (int8_t)(b1 - 128);
    }
}

// ---------------------------------------------------------------------------
// Reliable TCP send
// ---------------------------------------------------------------------------

static int send_all(int sock, const uint8_t* buf, size_t len)
{
    size_t sent = 0;
    while (sent < len) {
        int n = send(sock, buf + sent, len - sent, 0);
        if (n <= 0) return -1;
        sent += n;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// stream_task — Core 1
//
// Grabs YUV frames, encodes to JPEG for transmission, runs inference every
// INFER_EVERY_N_FRAMES frames and prints predictions to serial.
// Accepts one TCP client at a time; blocks until client disconnects then
// waits for the next connection.
// ---------------------------------------------------------------------------

static tflite::MicroInterpreter* s_interpreter  = nullptr;
static TfLiteTensor*             s_input_tensor = nullptr;

static void stream_task(void*)
{
    if (camera_init() != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed — halting stream_task");
        vTaskDelete(NULL);
        return;
    }

    /* Warm-up: OV3660 needs a few frames for AGC/AEC to stabilise */
    ESP_LOGI(TAG, "Camera warm-up...");
    for (int i = 0; i < 5; i++) {
        camera_fb_t* wb = esp_camera_fb_get();
        if (wb) esp_camera_fb_return(wb);
        vTaskDelay(pdMS_TO_TICKS(150));
    }
    ESP_LOGI(TAG, "Camera ready.");

    /* TCP server — one client at a time */
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(STREAM_PORT);
    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 1);

    ESP_LOGI(TAG, "Listening on port %d  (connect via %s.local)",
             STREAM_PORT, MDNS_HOSTNAME);

    uint32_t frame_n = 0;

    while (true) {
        struct sockaddr_in client_addr;
        socklen_t clen = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &clen);
        if (client_fd < 0) {
            ESP_LOGE(TAG, "accept() failed");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        ESP_LOGI(TAG, "Client connected from %s", inet_ntoa(client_addr.sin_addr));
        frame_n = 0;

        while (true) {
            camera_fb_t* fb = esp_camera_fb_get();
            if (!fb) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }

            /* --- Inference every N frames --------------------------------- */
            if (frame_n % INFER_EVERY_N_FRAMES == 0 && s_interpreter != nullptr) {
                int64_t t_inv = esp_timer_get_time();
                yuv422_to_int8_tensor(fb, s_input_tensor);

                if (s_interpreter->Invoke() == kTfLiteOk) {
                    TfLiteTensor* out = s_interpreter->output(0);

                    float powerbank_prob = 0.0f;
                    if (out->type == kTfLiteInt8) {
                        powerbank_prob =
                            (float)(out->data.int8[0] - (int)(out->params.zero_point))
                            * out->params.scale;
                    } else {
                        powerbank_prob = out->data.f[0];
                    }
                    powerbank_prob = powerbank_prob < 0.f ? 0.f :
                                     powerbank_prob > 1.f ? 1.f : powerbank_prob;

                    float guava_prob  = 1.0f - powerbank_prob;
                    bool  is_pb       = (powerbank_prob >= 0.5f);
                    float conf        = is_pb ? powerbank_prob : guava_prob;
                    float invoke_ms   = (esp_timer_get_time() - t_inv) / 1000.f;

                    /* Print exactly as main_tinyml_probe.cc does */
                    ESP_LOGI(TAG, "--- Frame %lu ---", (unsigned long)frame_n);
                    ESP_LOGI(TAG, "  guava     : %.4f  (%.1f%%)", guava_prob,     guava_prob     * 100.f);
                    ESP_LOGI(TAG, "  powerbank : %.4f  (%.1f%%)", powerbank_prob, powerbank_prob * 100.f);
                    ESP_LOGI(TAG, "  PREDICTION : %s  (%.1f%%)",
                             is_pb ? CLASS_LABELS[1] : CLASS_LABELS[0], conf * 100.f);
                    ESP_LOGI(TAG, "  invoke    : %.1f ms", invoke_ms);
                }
            }

            /* --- Encode YUV frame to JPEG for transmission --------------- */
            uint8_t* jpg_buf = NULL;
            size_t   jpg_len = 0;
            bool ok = frame2jpg(fb, 80, &jpg_buf, &jpg_len);
            esp_camera_fb_return(fb);

            if (!ok) {
                ESP_LOGW(TAG, "frame2jpg failed, skipping frame");
                frame_n++;
                continue;
            }

            /* --- Send: [4-byte LE length][JPEG bytes] -------------------- */
            uint32_t flen = (uint32_t)jpg_len;
            uint8_t header[4] = {
                (uint8_t)( flen        & 0xFF),
                (uint8_t)((flen >>  8) & 0xFF),
                (uint8_t)((flen >> 16) & 0xFF),
                (uint8_t)((flen >> 24) & 0xFF),
            };

            int err = send_all(client_fd, header, 4);
            if (err == 0) err = send_all(client_fd, jpg_buf, jpg_len);
            free(jpg_buf);

            if (err < 0) {
                ESP_LOGW(TAG, "Client disconnected");
                break;
            }

            frame_n++;
        }

        close(client_fd);
    }
}

// ---------------------------------------------------------------------------
// app_main
// ---------------------------------------------------------------------------

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "EdgeCV — Raw TCP Stream + Serial Inference");
    ESP_LOGI(TAG, "  SSID    : %s", WIFI_SSID);
    ESP_LOGI(TAG, "  Host    : %s.local:%d", MDNS_HOSTNAME, STREAM_PORT);
    ESP_LOGI(TAG, "  Model   : embedded (%u bytes)", model_tflite_len);
    ESP_LOGI(TAG, "  Infer   : every %d frames", INFER_EVERY_N_FRAMES);
    ESP_LOGI(TAG, "========================================");

    ESP_ERROR_CHECK(nvs_flash_init());

    /* --- TFLite ---------------------------------------------------------- */
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
    if (!arena) { ESP_LOGE(TAG, "ABORT: arena malloc"); return; }

    static tflite::MicroInterpreter interpreter(
        model, resolver, arena, TENSOR_ARENA_SIZE);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "ABORT: AllocateTensors failed"); return;
    }
    ESP_LOGI(TAG, "AllocateTensors OK — arena used %zu KB",
             interpreter.arena_used_bytes() / 1024);

    s_interpreter  = &interpreter;
    s_input_tensor = interpreter.input(0);

    if (s_input_tensor->type != kTfLiteInt8 ||
        (int)(s_input_tensor->dims->data[1]) != INPUT_H ||
        (int)(s_input_tensor->dims->data[2]) != INPUT_W ||
        (int)(s_input_tensor->dims->data[3]) != INPUT_C) {
        ESP_LOGE(TAG, "ABORT: input tensor mismatch");
        return;
    }
    ESP_LOGI(TAG, "Input tensor OK: int8[1,%d,%d,%d]  zp=%d",
             INPUT_H, INPUT_W, INPUT_C,
             (int)(s_input_tensor->params.zero_point));

    /* --- WiFi + mDNS ----------------------------------------------------- */
    wifi_init();
    mdns_init_service();

    /* --- Stream task on Core 1 ------------------------------------------- */
    xTaskCreatePinnedToCore(stream_task, "stream_task",
                            8192, NULL, 5, NULL, 1);

    /* Main task — keep alive, periodic heap report */
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(30000));
        ESP_LOGI(TAG, "heap=%zu KB  psram=%zu KB",
                 heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024,
                 heap_caps_get_free_size(MALLOC_CAP_SPIRAM)   / 1024);
    }
}
