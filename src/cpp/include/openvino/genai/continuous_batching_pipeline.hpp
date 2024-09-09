// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <openvino/openvino.hpp>

#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {
struct PipelineMetrics { 
    // All requests as viewed by the pipeline
    size_t requests = 0;
    // Requests scheduled for processing
    size_t scheduled_requests = 0;
    // Percentage of KV cache usage
    float cache_usage = 0.0;
};

class OPENVINO_GENAI_EXPORTS ContinuousBatchingPipeline {
    class Impl;
    std::shared_ptr<Impl> m_impl;

public:
    ContinuousBatchingPipeline() = default;

    ContinuousBatchingPipeline(const std::string& models_path,
                               const SchedulerConfig& scheduler_config,
                               const std::string& device = "CPU",
                               const ov::AnyMap& llm_plugin_config = {},
                               const ov::AnyMap& tokenizer_plugin_config = {});

    /**
    * @brief Constructs a ContinuousBatchingPipeline when ov::genai::Tokenizer is initialized manually using file from the different dirs.
    *
    * @param model_path Path to the dir with model, tokenizer .xml/.bin files, and generation_configs.json
    * @param scheduler_config
    * @param tokenizer manually initialized ov::genai::Tokenizer
    * @param device optional device
    * @param plugin_config optional plugin_config
    */
    ContinuousBatchingPipeline(
        const std::string& model_path,
        const ov::genai::Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device="CPU",
        const ov::AnyMap& plugin_config={}
    );

    ContinuousBatchingPipeline(
        ov::Core& core,
        const std::shared_ptr<ov::Model>& model,
        const ov::genai::Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device="CPU",
        const ov::AnyMap& plugin_config={},
        bool is_enable_validation_mode=false
    );

    ov::genai::Tokenizer get_tokenizer();

    ov::genai::GenerationConfig get_config() const;

    PipelineMetrics get_metrics() const;

    GenerationHandle add_request(uint64_t request_id, const ov::Tensor& input_ids, const ov::genai::GenerationConfig& sampling_params);
    GenerationHandle add_request(uint64_t request_id, const std::string& prompt, const ov::genai::GenerationConfig& sampling_params);

    void step();

    bool has_non_finished_requests();

    // more high level interface, which can process multiple prompts in continuous batching manner
    std::vector<EncodedGenerationResult> generate(const std::vector<ov::Tensor>& input_ids, const std::vector<ov::genai::GenerationConfig>& sampling_params, const ov::genai::StreamerVariant& streamer=std::monostate{});
    std::vector<GenerationResult> generate(const std::vector<std::string>& prompts, const std::vector<ov::genai::GenerationConfig>& sampling_params, const ov::genai::StreamerVariant& streamer=std::monostate{});

    /**
    * @brief start chat with keeping history in kv cache.
    *
    * @param system_message optional system message.
    */
    void start_chat(const std::string& system_message = "");

    /**
    * @brief finish chat and clear kv cache.
    */
    void finish_chat();

    // for speculative decoding
    void finish_request(int64_t request_id = -1);

    struct GeneratedSequence {
        uint64_t request_id = 0, sequence_id = 0;
        std::vector<int64_t> token_ids;
        std::vector<float> log_probs;

        GeneratedSequence(uint64_t req_id, uint64_t seq_id, const  std::vector<int64_t>& generated_token_ids, const std::vector<float>& generated_log_probs) :
            request_id(req_id),
            sequence_id(seq_id),
            token_ids(generated_token_ids),
            log_probs(generated_log_probs) {};
    };

    struct UpdateSeqResult {
        size_t to_insert, to_remove;
        UpdateSeqResult(size_t _to_insert = 0, size_t _to_remove = 0) : to_insert(_to_insert), to_remove(_to_remove) {};
    };

    std::vector<GeneratedSequence> get_generated_sequences();
    UpdateSeqResult update_generated_sequence(const GeneratedSequence& new_sequence);
};

std::shared_ptr<ov::Model> OPENVINO_GENAI_EXPORTS read_model_and_apply_paged_attention(const std::string& model_path, ov::Core& core);

}
