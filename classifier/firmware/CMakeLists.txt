cmake_minimum_required(VERSION 3.16)

# Both defaults files are merged, in order. This used to be unset, which meant
# idf.py silently loaded only sdkconfig.defaults and ignored
# sdkconfig.defaults.camera.additions entirely — its settings had never
# actually reached a build, despite the README instructing you to merge them.
set(SDKCONFIG_DEFAULTS
    "sdkconfig.defaults;sdkconfig.defaults.camera.additions")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(edgecv_classifier)
