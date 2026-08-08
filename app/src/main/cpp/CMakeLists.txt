cmake_minimum_required(VERSION 3.22.1)
project(pointcloud3dgs LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(VKSPLAT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vksplat/vksplat)
set(VKSPLAT_SRC ${VKSPLAT_ROOT}/src)
set(GLM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm)

if(NOT EXISTS ${VKSPLAT_SRC}/gs_trainer.cpp)
    message(FATAL_ERROR "VkSplat source missing. Run scripts/prepare-vksplat-android.sh before Gradle build.")
endif()
if(NOT EXISTS ${GLM_ROOT}/glm/glm.hpp)
    message(FATAL_ERROR "GLM source missing. Run scripts/prepare-vksplat-android.sh before Gradle build.")
endif()

add_library(pointcloud3dgs SHARED
    native_3dgs.cpp
    ${VKSPLAT_SRC}/buffer.cpp
    ${VKSPLAT_SRC}/colmap_reader.cpp
    ${VKSPLAT_SRC}/knn.cpp
    ${VKSPLAT_SRC}/scheduler.cpp
    ${VKSPLAT_SRC}/gs_pipeline.cpp
    ${VKSPLAT_SRC}/gs_renderer.cpp
    ${VKSPLAT_SRC}/gs_trainer.cpp
    ${VKSPLAT_SRC}/perf_timer.cpp
)

target_include_directories(pointcloud3dgs PRIVATE
    ${VKSPLAT_SRC}
    ${GLM_ROOT}
)

target_compile_definitions(pointcloud3dgs PRIVATE
    VK_USE_PLATFORM_ANDROID_KHR=1
    SUBGROUP_SIZE=32
)

target_compile_options(pointcloud3dgs PRIVATE
    -O3
    -fexceptions
    -frtti
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-sign-compare
)

find_library(log-lib log)
find_library(android-lib android)
find_library(vulkan-lib vulkan)

target_link_libraries(pointcloud3dgs PRIVATE
    ${log-lib}
    ${android-lib}
    ${vulkan-lib}
)
