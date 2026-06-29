# classifier/firmware/main/CMakeLists.txt
#
# BUILD_MODE selects which entry point gets compiled.
#
#   probe   (default) — main_tinyml_probe.cc  — serial-only inference, no WiFi
#   stream              main_stream_raw.cc     — TCP raw stream + serial inference + mDNS
#   wifi                main_wifi_stream.cc    — WiFi + MJPEG browser stream + inference
#
# Usage:
#   idf.py build                                          # probe (default)
#   idf.py -DBUILD_MODE=stream build flash monitor        # TCP stream to laptop
#   idf.py -DBUILD_MODE=wifi   build flash monitor        # browser MJPEG stream
#
# Legacy APP_MAIN env-var still works if BUILD_MODE is not set, for backwards
# compatibility. BUILD_MODE takes precedence when both are present.

cmake_minimum_required(VERSION 3.16)

# --- resolve BUILD_MODE ---
if(NOT DEFINED BUILD_MODE OR BUILD_MODE STREQUAL "")
    # fall back to legacy APP_MAIN env var if set
    if(DEFINED ENV{APP_MAIN} AND NOT "$ENV{APP_MAIN}" STREQUAL "")
        set(BUILD_MODE_RESOLVED "$ENV{APP_MAIN}")
        message(STATUS "EdgeCV: BUILD_MODE not set, using legacy APP_MAIN=$ENV{APP_MAIN}")
    else()
        set(BUILD_MODE_RESOLVED "probe")
        message(STATUS "EdgeCV: BUILD_MODE not set, defaulting to 'probe'")
    endif()
else()
    set(BUILD_MODE_RESOLVED "${BUILD_MODE}")
    message(STATUS "EdgeCV: BUILD_MODE=${BUILD_MODE}")
endif()

# --- map mode name → source file ---
if(BUILD_MODE_RESOLVED STREQUAL "stream" OR BUILD_MODE_RESOLVED STREQUAL "main_stream_raw.cc")
    set(MAIN_SRC "main_stream_raw.cc")
    message(STATUS "EdgeCV: building TCP stream entry point (main_stream_raw.cc)")

elseif(BUILD_MODE_RESOLVED STREQUAL "wifi" OR BUILD_MODE_RESOLVED STREQUAL "main_wifi_stream.cc")
    set(MAIN_SRC "main_wifi_stream.cc")
    message(STATUS "EdgeCV: building WiFi/MJPEG entry point (main_wifi_stream.cc)")

elseif(BUILD_MODE_RESOLVED STREQUAL "probe" OR BUILD_MODE_RESOLVED STREQUAL "main_tinyml_probe.cc")
    set(MAIN_SRC "main_tinyml_probe.cc")
    message(STATUS "EdgeCV: building serial-only probe entry point (main_tinyml_probe.cc)")

elseif(BUILD_MODE_RESOLVED STREQUAL "edgecv_v2" OR BUILD_MODE_RESOLVED STREQUAL "main_edgecv_v2.cc")
    set(MAIN_SRC "main_edgecv_v2.cc")
    message(STATUS "EdgeCV: building Phase 2 profiling+rejection entry point (main_edgecv_v2.cc)")

elseif(BUILD_MODE_RESOLVED STREQUAL "model_probe" OR BUILD_MODE_RESOLVED STREQUAL "main_model_probe.cc")
    set(MAIN_SRC "main_model_probe.cc")
    message(STATUS "EdgeCV: building model-load probe (main_model_probe.cc)")

elseif(BUILD_MODE_RESOLVED STREQUAL "camera_probe" OR BUILD_MODE_RESOLVED STREQUAL "main_camera_probe.cc")
    set(MAIN_SRC "main_camera_probe.cc")
    message(STATUS "EdgeCV: building camera-coexistence probe (main_camera_probe.cc)")

else()
    # treat as raw filename for forward-compat
    set(MAIN_SRC "${BUILD_MODE_RESOLVED}")
    message(WARNING "EdgeCV: unrecognised BUILD_MODE '${BUILD_MODE_RESOLVED}', using as filename")
endif()

idf_component_register(
    SRCS
        "${MAIN_SRC}"
        "model_data.cc"
    INCLUDE_DIRS "."
    REQUIRES
        esp_psram
        esp_timer
        esp32-camera
        esp_wifi
        esp_event
        esp_netif
        nvs_flash
        mdns
        driver
        log
        esp_http_server
)
