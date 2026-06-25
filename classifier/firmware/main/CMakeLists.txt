## main/CMakeLists.txt — classifier/firmware/main
##
## Entry points (select with APP_MAIN env var before idf.py build):
##   main_tinyml_probe.cc  — standalone inference, serial output only (DEFAULT)
##   main_wifi_stream.cc   — inference + MJPEG live stream + browser UI
##   main_model_probe.cc   — diagnostic: model load + AllocateTensors only
##   main_camera_probe.cc  — diagnostic: camera + model coexistence check
##
## Examples:
##   idf.py build flash monitor                              # tinyml_probe
##   APP_MAIN=main_wifi_stream.cc idf.py build flash monitor # wifi stream

if(DEFINED ENV{APP_MAIN})
    set(EDGECV_APP_MAIN $ENV{APP_MAIN})
else()
    set(EDGECV_APP_MAIN "main_tinyml_probe.cc")
endif()

message(STATUS "EdgeCV: building ${EDGECV_APP_MAIN}")

idf_component_register(
    SRCS
        "${EDGECV_APP_MAIN}"
        "model_data.cc"
    INCLUDE_DIRS "."
    REQUIRES
        esp_psram
        esp_timer
        driver
        log
        heap
        spi_flash
        esp32-camera
        esp_wifi
        esp_event
        esp_netif
        esp_http_server
        nvs_flash
)
