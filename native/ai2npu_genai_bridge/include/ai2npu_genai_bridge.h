#pragma once

#include <stddef.h>
#include <stdbool.h>

#ifdef _WIN32
#define AI2NPU_BRIDGE_API __declspec(dllexport)
#else
#define AI2NPU_BRIDGE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ai2npu_whisper_t ai2npu_whisper_t;

AI2NPU_BRIDGE_API int ai2npu_whisper_create(
    const char* model_dir,
    const char* device,
    ai2npu_whisper_t** out,
    char** error);

AI2NPU_BRIDGE_API int ai2npu_whisper_transcribe(
    ai2npu_whisper_t* handle,
    const float* samples,
    size_t sample_count,
    const char* task,
    const char* language,
    const char* prompt,
    float temperature,
    bool return_timestamps,
    char** json_out,
    char** error);

AI2NPU_BRIDGE_API void ai2npu_whisper_free(ai2npu_whisper_t* handle);
AI2NPU_BRIDGE_API void ai2npu_whisper_free_string(char* value);

#ifdef __cplusplus
}
#endif
