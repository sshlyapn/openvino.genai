// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <cxxopts.hpp>

#include "openvino/genai/generation_config.hpp"
#include "speculative_decoding_pipeline.hpp"

void print_generation_result(const ov::genai::GenerationResult& generation_result) {
    for (size_t output_id = 0; output_id < generation_result.m_generation_ids.size(); ++output_id) {
        std::cout << "Answer " << output_id << " (" << generation_result.m_scores[output_id] << ") : " << generation_result.m_generation_ids[output_id] << std::endl;
    }
}

int main(int argc, char* argv[]) try {
    // Command line options

    cxxopts::Options options("accuracy_sample", "Help command");

    options.add_options()
    ("n,num_prompts", "A number of prompts", cxxopts::value<size_t>()->default_value("1"))
    ("dynamic_split_fuse", "Whether to use dynamic split-fuse or vLLM scheduling", cxxopts::value<bool>()->default_value("false"))
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("a,assisting_model", "Path to assisting model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("k,candidates_number", "candidates_number", cxxopts::value<size_t>()->default_value("5"))
    ("g,generated_len", "generated_len", cxxopts::value<size_t>()->default_value("30"))
    ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    const size_t num_prompts = result["num_prompts"].as<size_t>();
    const bool dynamic_split_fuse = result["dynamic_split_fuse"].as<bool>();
    const std::string models_path = result["model"].as<std::string>();
    const std::string assisting_models_path = result["assisting_model"].as<std::string>();
    const size_t k = result["candidates_number"].as<size_t>();
    const size_t g = result["generated_len"].as<size_t>();

    // create dataset

    std::vector<std::string> prompt_examples = {
        "The United Arab Emirates[c] (UAE), or simply the Emirates,[d] is a country in West Asia, in the Middle East, at the eastern end of the Arabian Peninsula. It is a federal, elective monarchy composed of seven emirates, with Abu Dhabi as its capital.[13] It shares land borders with Oman to the east and northwest, and with Saudi Arabia to the southwest; as well as maritime borders in the Persian Gulf with Qatar and Iran, and with Oman in the Gulf of Oman.",
        "What is OpenVINO?",
        "How are you?",
        "What is your name?",
        "Tell me something about Canada",
        "What is OpenVINO?",
    };

    auto greedy = ov::genai::greedy();
    greedy.max_new_tokens = g;

    std::vector<ov::genai::GenerationConfig> sampling_params_examples {
        // ov::genai::beam_search(),
        greedy,
        // ov::genai::multinomial(),
    };

    std::vector<std::string> prompts(num_prompts);
    std::vector<ov::genai::GenerationConfig> sampling_params(num_prompts);

    for (size_t request_id = 0; request_id < num_prompts; ++request_id) {
        prompts[request_id] = prompt_examples[request_id % prompt_examples.size()];
        sampling_params[request_id] = sampling_params_examples[request_id % sampling_params_examples.size()];
    }

    // Perform the inference

    ov::genai::SchedulerConfig scheduler_config;
    // batch size
    scheduler_config.max_num_batched_tokens = 32;
    // cache params
    scheduler_config.num_kv_blocks = 364;
    scheduler_config.block_size = 32;
    // mode - vLLM or dynamic_split_fuse
    scheduler_config.dynamic_split_fuse = dynamic_split_fuse;
    // vLLM specific params
    scheduler_config.max_num_seqs = 2;

    // It's possible to construct a Tokenizer from a different path.
    // If the Tokenizer isn't specified, it's loaded from the same folder.
    SpeculativeDecodingPipeline pipe(models_path, assisting_models_path, k, scheduler_config, "CPU");
    std::vector<ov::genai::GenerationResult> generation_results = pipe.generate(prompts, sampling_params);

    for (size_t request_id = 0; request_id < generation_results.size(); ++request_id) {
        const ov::genai::GenerationResult & generation_result = generation_results[request_id];
        std::cout << "Question: " << prompts[request_id] << std::endl;
        switch (generation_result.m_status)
        {
        case ov::genai::GenerationStatus::FINISHED:
            print_generation_result(generation_result);
            break;
        case ov::genai::GenerationStatus::IGNORED:
            std::cout << "Request was ignored due to lack of memory." <<std::endl;
            if (generation_result.m_generation_ids.size() > 0) {
                std::cout << "Partial result:" << std::endl;
                print_generation_result(generation_result);
            }
            break;
        case ov::genai::GenerationStatus::DROPPED_BY_PIPELINE:
            std::cout << "Request was aborted." <<std::endl;
            if (generation_result.m_generation_ids.size() > 0) {
                std::cout << "Partial result:" << std::endl;
                print_generation_result(generation_result);
            }
            break;   
        default:
            break;
        }
        std::cout << std::endl;
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}