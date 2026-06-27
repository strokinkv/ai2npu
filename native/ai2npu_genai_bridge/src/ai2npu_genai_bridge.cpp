#include "ai2npu_genai_bridge.h"

#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/runtime/properties.hpp"

#include <cstring>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

struct ai2npu_whisper_t {
    std::unique_ptr<ov::genai::WhisperPipeline> pipeline;
};

namespace {

char* copy_string(const std::string& value) {
    auto* out = new char[value.size() + 1];
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

void trace(const std::string& message) {
    const char* enabled = std::getenv("AI2NPU_BRIDGE_TRACE");
    if (enabled != nullptr && enabled[0] == '1') {
        std::cerr << "[ai2npu_genai_bridge] " << message << std::endl;
    }
}

int fail(char** error, const std::string& message) {
    if (error != nullptr) {
        *error = copy_string(message);
    }
    return 1;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                out << "\\u00";
                const char* hex = "0123456789abcdef";
                out << hex[(ch >> 4) & 0x0f] << hex[ch & 0x0f];
            } else {
                out << ch;
            }
            break;
        }
    }
    return out.str();
}

std::string language_json(const char* language) {
    if (language == nullptr || language[0] == '\0') {
        return "null";
    }
    return "\"" + json_escape(language) + "\"";
}

std::string joined_text(const ov::genai::WhisperDecodedResults& result) {
    std::ostringstream out;
    for (const auto& text : result.texts) {
        out << text;
    }
    return out.str();
}

std::string result_to_json(const ov::genai::WhisperDecodedResults& result, const char* language, double duration_sec) {
    std::ostringstream out;
    out << "{\"text\":\"" << json_escape(joined_text(result)) << "\",";
    out << "\"language\":" << language_json(language) << ",";
    out << "\"duration\":" << duration_sec << ",";
    out << "\"segments\":[";
    if (result.chunks.has_value()) {
        bool first = true;
        for (const auto& chunk : *result.chunks) {
            if (!first) {
                out << ",";
            }
            first = false;
            out << "{\"start\":" << chunk.start_ts << ",";
            out << "\"end\":" << chunk.end_ts << ",";
            out << "\"text\":\"" << json_escape(chunk.text) << "\"}";
        }
    }
    out << "]}";
    return out.str();
}

ov::AnyMap ov_config_for_device(const std::string& device) {
    ov::AnyMap ov_config;
    if (device != "NPU") {
        return ov_config;
    }

    const char* cache_dir = std::getenv("AI2NPU_WHISPER_CACHE_DIR");
    if (cache_dir != nullptr && cache_dir[0] != '\0') {
        ov_config.insert({ov::cache_dir(cache_dir)});
    }
    return ov_config;
}

size_t max_new_tokens_from_env() {
    const char* value = std::getenv("AI2NPU_WHISPER_MAX_NEW_TOKENS");
    if (value == nullptr || value[0] == '\0') {
        return 448;
    }
    try {
        return static_cast<size_t>(std::stoul(value));
    } catch (...) {
        return 448;
    }
}

} // namespace

extern "C" AI2NPU_BRIDGE_API int ai2npu_whisper_create(
    const char* model_dir,
    const char* device,
    ai2npu_whisper_t** out,
    char** error) {
    if (out == nullptr) {
        return fail(error, "out pointer is null");
    }
    *out = nullptr;
    if (model_dir == nullptr || device == nullptr) {
        return fail(error, "model_dir and device are required");
    }

    try {
        std::string requested_device(device);
        if (requested_device != "NPU") {
            return fail(error, "only NPU device is supported");
        }

        auto handle = std::make_unique<ai2npu_whisper_t>();
        trace("creating WhisperPipeline");
        handle->pipeline = std::make_unique<ov::genai::WhisperPipeline>(
            std::filesystem::path(model_dir),
            requested_device,
            ov_config_for_device(requested_device));
        trace("WhisperPipeline created");
        *out = handle.release();
        return 0;
    } catch (const std::exception& ex) {
        return fail(error, ex.what());
    } catch (...) {
        return fail(error, "unknown exception while creating Whisper pipeline");
    }
}

extern "C" AI2NPU_BRIDGE_API int ai2npu_whisper_transcribe(
    ai2npu_whisper_t* handle,
    const float* samples,
    size_t sample_count,
    const char* task,
    const char* language,
    const char* prompt,
    float temperature,
    bool return_timestamps,
    char** json_out,
    char** error) {
    if (json_out == nullptr) {
        return fail(error, "json_out pointer is null");
    }
    *json_out = nullptr;
    if (handle == nullptr || handle->pipeline == nullptr) {
        return fail(error, "Whisper pipeline handle is null");
    }
    if (samples == nullptr && sample_count > 0) {
        return fail(error, "samples pointer is null");
    }

    try {
        ov::genai::RawSpeechInput raw_speech(samples, samples + sample_count);
        auto config = handle->pipeline->get_generation_config();
        config.task = task != nullptr ? task : "transcribe";
        config.return_timestamps = return_timestamps;
        config.word_timestamps = false;
        config.max_new_tokens = max_new_tokens_from_env();
        if (temperature >= 0.0f) {
            config.temperature = temperature;
        }
        if (language != nullptr && language[0] != '\0') {
            std::string token = language;
            if (token.rfind("<|", 0) != 0) {
                token = "<|" + token + "|>";
            }
            config.language = token;
        }
        // NOTE: `initial_prompt` is deliberately NOT forwarded on NPU. The
        // static-shape NPU Whisper pipeline hangs inside `generate()` when an
        // initial/context prompt is supplied (the decoder cannot grow its
        // cross-attention prefix), so a second streaming phrase — conditioned on
        // the first phrase's transcript — would never return. Like
        // `word_timestamps` above, this is a hard NPU limitation neutralised at
        // the device boundary. `prompt` is accepted for ABI/source compatibility
        // but intentionally ignored.
        (void)prompt;

        trace("starting WhisperPipeline::generate");
        auto result = handle->pipeline->generate(raw_speech, config);
        trace("WhisperPipeline::generate finished");
        const double duration_sec = static_cast<double>(sample_count) / 16000.0;
        *json_out = copy_string(result_to_json(result, language, duration_sec));
        return 0;
    } catch (const std::exception& ex) {
        return fail(error, ex.what());
    } catch (...) {
        return fail(error, "unknown exception while running Whisper transcription");
    }
}

extern "C" AI2NPU_BRIDGE_API void ai2npu_whisper_free(ai2npu_whisper_t* handle) {
    delete handle;
}

extern "C" AI2NPU_BRIDGE_API void ai2npu_whisper_free_string(char* value) {
    delete[] value;
}
