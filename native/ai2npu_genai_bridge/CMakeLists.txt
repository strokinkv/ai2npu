cmake_minimum_required(VERSION 3.24)
project(ai2npu_genai_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

add_library(ai2npu_genai_bridge SHARED
    src/ai2npu_genai_bridge.cpp
)

target_include_directories(ai2npu_genai_bridge
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(OpenVINOGenAI REQUIRED)

target_link_libraries(ai2npu_genai_bridge
    PRIVATE
        openvino::genai
)

set_target_properties(ai2npu_genai_bridge PROPERTIES
    OUTPUT_NAME ai2npu_genai_bridge
)
