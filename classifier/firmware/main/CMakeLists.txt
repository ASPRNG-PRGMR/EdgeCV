## CMakeLists.txt — classifier/firmware
##
## This single project contains three entry points (main_*.cc), one for
## each deployment stage documented in training_devlog.md Phase 9/10:
##
##   main_model_probe.cc   — Probe 1: model load + AllocateTensors only
##   main_camera_probe.cc  — Probe 2: model + camera coexistence check
##   main_inference.cc     — full pipeline (the deployed application)
##
## Only one is compiled per build. Select it with the APP_MAIN environment
## variable before calling idf.py build. If unset, main_inference.cc is
## used — so a plain `idf.py build flash monitor` always builds the real
## deployed application.
##
## Examples:
##   APP_MAIN=main_model_probe.cc  idf.py build flash monitor
##   APP_MAIN=main_camera_probe.cc idf.py build flash monitor
##   idf.py build flash monitor                # builds main_inference.cc

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
)
