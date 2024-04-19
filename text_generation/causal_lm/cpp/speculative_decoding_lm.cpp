// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <cmath>
#include <random>

constexpr size_t BATCH_SIZE = 1;

// sequence length axis in key/values tensors, for most cases [BATCH_SIZE, num_kv_heads, seq_len, head_size],
// threfore usually SEQ_LEN_AXIS = 2
constexpr size_t SEQ_LEN_AXIS = 2;

// There's no way to extract special token values from the detokenizer for now
constexpr int64_t SPECIAL_EOS_TOKEN = 2;

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string&& prompt) {
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t>& tokens) {
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    ov::InferRequest detokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = detokenize(detokenizer, token_cache);
        if (!text.empty() && '\n' == text.back()) {
            // Flush the cache after the new line symbol
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
            token_cache.clear();
            print_len = 0;
	    return;
        }
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // Don't print incomplete text
            return;
        }
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
        print_len = text.size();
    }

    void end() {
        std::string text = detokenize(detokenizer, token_cache);
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};
}

ov::Tensor trimm_tensor(ov::Tensor& tensor, uint64_t seq_len_axis, uint64_t new_seq_len) {
    // Copy elements from the old to a new tensor and return it.
    // It's assumed that key/values tensor has a shape [BATCH_SIZE, num_kv_heads, seq_len, head_size] or [seq_len, ...],
    // It that's not the case for your model please implement your own trim method.
    OPENVINO_ASSERT(seq_len_axis == 2 || seq_len_axis == 0, "Cannot trim key/values with sequence length axis = ", seq_len_axis);

    auto old_tensor_data = tensor.data<float>();
    auto shape = tensor.get_shape();
    size_t num_kv_heads = shape[1];
    size_t old_seq_len = shape[2];
    size_t head_size = shape[3];

    OPENVINO_ASSERT(new_seq_len <= old_seq_len);

    // if new_seq_len equal to old one no need to copy tensor, return as is
    if (old_seq_len == new_seq_len)
        return tensor;

    if (seq_len_axis == 0) {
        shape[0] = new_seq_len;
        tensor.set_shape(shape);
    }

    // if seq_len_axis == 2, then data is not contiguous, in order to trim need to repack tensor
    auto new_tensor = ov::Tensor{ov::element::f32, {BATCH_SIZE, num_kv_heads, new_seq_len, head_size}};
    auto new_tensor_data = new_tensor.data<float>();
    for (size_t batch = 0; batch < BATCH_SIZE; ++batch){
        for (size_t i = 0; i < num_kv_heads; ++i) {
            for (size_t j = 0; j < new_seq_len; ++j) {
                auto dst_ptr = new_tensor_data + num_kv_heads * new_seq_len * head_size * batch + new_seq_len * head_size * i +  head_size * j;
                auto src_ptr = old_tensor_data + num_kv_heads * new_seq_len * head_size * batch + old_seq_len * head_size * i +  head_size * j;
                std::memcpy(dst_ptr, src_ptr, head_size * sizeof(float));
            }
        }
    }
    return new_tensor;
}

void update_kv_cache(ov::InferRequest request, uint64_t seq_len_axis, uint64_t new_seq_len) {
    // trim kv_cache values up to the new_seq_len
    for (auto& state: request.query_state()) {
        ov::Tensor old_tensor = state.get_state();
        state.set_state(trimm_tensor(old_tensor, seq_len_axis, new_seq_len));
    }
}

struct PagedAttentionManager {
public:
    PagedAttentionManager(ov::Tensor& slot_mapping, ov::Tensor& max_context_len, ov::Tensor& context_lens, ov::Tensor& block_tables, size_t block_size = 16)
        : slot_mapping(slot_mapping),
          max_context_len(max_context_len),
          context_lens(context_lens),
          block_tables(block_tables),
          block_size(block_size) {}

    void update_tensors(const ov::Tensor& input_ids) {
        int64_t prev_seq_len = seq_len;
        seq_len += input_ids.get_shape()[1];

        // std::cout << "Updated seq_len: new " << seq_len << " prev " << prev_seq_len << "\n";

        slot_mapping.set_shape(input_ids.get_shape());
        std::iota(slot_mapping.data<int64_t>(), slot_mapping.data<int64_t>() + seq_len - prev_seq_len, prev_seq_len);

        max_context_len.data<int32_t>()[0] = seq_len;

        context_lens.set_shape({BATCH_SIZE});
        context_lens.data<int64_t>()[0] = seq_len;

        size_t blocks_num = (seq_len + block_size - 1) / block_size;
        block_tables.set_shape({BATCH_SIZE, blocks_num});
        std::iota(block_tables.data<int32_t>(), block_tables.data<int32_t>() + blocks_num, 0);
    }

    void reduce_seq_len(int64_t val) {
        seq_len -= val;
    }

private:
    ov::Tensor& slot_mapping;
    ov::Tensor& max_context_len;
    ov::Tensor& context_lens;
    ov::Tensor& block_tables;

    int64_t seq_len = 0;
    size_t block_size = 16;
};

int main(int argc, char* argv[]) try {
    if (argc != 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <DEVICE> <DRAFT MODEL_DIR> <MAIN MODEL_DIR> '<PROMPT>'");
    }

    // tokenizer model
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    core.add_extension("libuser_ov_extensions.so");
    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer = core.compile_model(
        std::string{argv[2]} + "/openvino_tokenizer.xml", "CPU").create_infer_request();
    auto [draft_input_ids, draft_attention_mask] = tokenize(tokenizer, argv[4]);

    ov::InferRequest detokenizer = core.compile_model(
        std::string{argv[2]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    TextStreamer text_streamer{std::move(detokenizer)};

    std::cout << "Tokenizer and detokenizer were loaded\n";

    std::string device = argv[1];
    auto draft_ov_model = core.read_model(std::string{argv[2]} + "/openvino_model.xml");

    auto draft_compiled_model = core.compile_model(draft_ov_model, device);

    std::cout << "Draft model [" << std::string{argv[2]} + "/openvino_model.xml" << " was loaded on " << device << "\n";
    ov::InferRequest draft_model = draft_compiled_model.create_infer_request();

    draft_model.set_tensor("input_ids", draft_input_ids);

    std::vector<ov::Tensor> draft_kv_inputs;
    std::vector<ov::Tensor> main_kv_inputs;

    // const size_t x_block_size = 16 / (cache_dt.bitwidth() / 8);
    const size_t x_block_size = 8;
    const size_t block_size = 16;

    std::cout << "Used block_size " << block_size << " x_block_size " << x_block_size << "\n";

    // Allocate inputs
    size_t kv_inputs_num = 20;
    for (size_t i = 0; i < kv_inputs_num * 2; i++) {
        // parameter:past_key_values.0.key was: f16:bfzyx:?x?x?x?x?:nopad now: f16:bfzyx:2416x32x16x16x8:nopad
        // parameter:past_key_values.0.value was: f16:bfyx:?x?x?x?:nopad now: f16:bfyx:2416x32x128x16:nopad
        const ov::element::Type cache_dt = draft_ov_model->input("past_key_values.0.key").get_element_type();
        const size_t kv_cache_blocks_num = 200;
        const size_t kv_heads_num = 16;
        const size_t head_size = 64;
        auto remote_context = draft_compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();

        ov::Shape key_cache_shape = {kv_cache_blocks_num, kv_heads_num, head_size / x_block_size, block_size, x_block_size};
        ov::Shape value_cache_shape = {kv_cache_blocks_num, kv_heads_num, head_size, block_size};

        if (i == 0) {
            std::cout << "Draft model key/value cache dt: " << cache_dt << "\n";
            std::cout << "Draft model key cache shape: " << key_cache_shape << "\n";
            std::cout << "Draft model value shape: " << value_cache_shape << "\n";
        }

        draft_kv_inputs.push_back(remote_context.create_tensor(cache_dt, i % 2 == 0 ? key_cache_shape : value_cache_shape));
    }

    ov::Tensor draft_position_ids = draft_model.get_tensor("position_ids");
    draft_position_ids.set_shape(draft_input_ids.get_shape());
    std::iota(draft_position_ids.data<int64_t>(), draft_position_ids.data<int64_t>() + draft_position_ids.get_size(), 0);
    uint64_t seq_len = draft_input_ids.get_shape()[1];

    // main model
    std::shared_ptr<ov::Model> main_ov_model;
    try {
        main_ov_model = core.read_model(std::string{argv[3]} + "/openvino_model.xml");
    } catch (const std::exception& ex) {
        std::cout << ex.what() << "\n";
    }

    bool fp32_for_main_model = true;
    if (fp32_for_main_model)
        std::cout << "\nINFO:fp32 was forced for main model\n\n";

    auto main_compiled_model = fp32_for_main_model ? core.compile_model(main_ov_model, device, ov::hint::inference_precision(ov::element::f32))
                                                   : core.compile_model(main_ov_model, device);

    ov::InferRequest main_model = main_compiled_model.create_infer_request();

    std::cout << "Main model [" << std::string{argv[3]} + "/openvino_model.xml" << " was loaded on " << device << "\n";

    // Allocate inputs
    kv_inputs_num = 32;
    for (size_t i = 0; i < kv_inputs_num * 2; i++) {
        // parameter:past_key_values.4.key was: f16:bfzyx:?x?x?x?x?:nopad now: f16:bfzyx:974x40x16x16x8:nopad
        // parameter:past_key_values.4.value was: f16:bfyx:?x?x?x?:nopad now: f16:bfyx:974x40x128x16:nopad
        const ov::element::Type cache_dt = main_ov_model->input("past_key_values.0.key").get_element_type();
        const size_t kv_cache_blocks_num = 200;
        const size_t kv_heads_num = 32;
        const size_t head_size = 80;
        auto remote_context = draft_compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();

        ov::Shape key_cache_shape = {kv_cache_blocks_num, kv_heads_num, head_size / x_block_size, block_size, x_block_size};
        ov::Shape value_cache_shape = {kv_cache_blocks_num, kv_heads_num, head_size, block_size};

        if (i == 0) {
            std::cout << "Draft model key/value cache dt: " << cache_dt << "\n";
            std::cout << "Draft model key cache shape: " << key_cache_shape << "\n";
            std::cout << "Draft model value shape: " << value_cache_shape << "\n";
        }

        main_kv_inputs.push_back(remote_context.create_tensor(cache_dt, i % 2 == 0 ? key_cache_shape : value_cache_shape));
    }

    // Input tensors for the main model should not be mixed with draft.
    // Do not feed the same draft_postion_ids to the main, but copy input_ids from the draft_input_ids
    auto input_ids = main_model.get_tensor("input_ids");
    input_ids.set_shape(draft_input_ids.get_shape());
    draft_input_ids.copy_to(input_ids);

    auto position_ids = main_model.get_tensor("position_ids");
    position_ids.set_shape(draft_input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);

    ov::Tensor draft_is_prompt = draft_model.get_tensor("is_prompt");
    draft_is_prompt.data<bool>()[0] = true;

    ov::Tensor is_prompt = main_model.get_tensor("is_prompt");
    is_prompt.data<bool>()[0] = true;

    ov::Tensor draft_slot_mapping = draft_model.get_tensor("slot_mapping");
    ov::Tensor draft_max_context_len = draft_model.get_tensor("max_context_len");
    ov::Tensor draft_context_lens = draft_model.get_tensor("context_lens");
    ov::Tensor draft_block_tables = draft_model.get_tensor("block_tables");

    ov::Tensor slot_mapping = main_model.get_tensor("slot_mapping");
    ov::Tensor max_context_len = main_model.get_tensor("max_context_len");
    ov::Tensor context_lens = main_model.get_tensor("context_lens");
    ov::Tensor block_tables = main_model.get_tensor("block_tables");

    PagedAttentionManager draft_model_pa_manager(draft_slot_mapping, draft_max_context_len, draft_context_lens, draft_block_tables);
    PagedAttentionManager main_model_pa_manager(slot_mapping, max_context_len, context_lens, block_tables);

    draft_model_pa_manager.update_tensors(draft_input_ids);
    main_model_pa_manager.update_tensors(input_ids);

    std::cout << "Set kv_cache for draft model (" << draft_kv_inputs.size() << ")\n";
    for (size_t i = 0; i < draft_kv_inputs.size(); i++) {
        std::string kv_input = "past_key_values." + std::to_string(i / 2) + "." + (i % 2 == 0 ? "key" : "value");
        draft_model.set_tensor(kv_input, draft_kv_inputs[i]);
    }

    std::cout << "Set kv_cache for main model (" << main_kv_inputs.size() << ")\n";
    for (size_t i = 0; i < main_kv_inputs.size(); i++) {
        std::string kv_input = "past_key_values." + std::to_string(i / 2) + "." + (i % 2 == 0 ? "key" : "value");
        main_model.set_tensor(kv_input, main_kv_inputs[i]);
    }

    std::cout << "Start inference \n";
    // set beam_idx for stateful model: no beam search is used and BATCH_SIZE = 1
    // draft_model.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    // draft_model.get_tensor("beam_idx").data<int32_t>()[0] = 0;
    // main_model.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    // main_model.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    // To coollect kv-cache for the <PROMPT> and to get the next token run the very first infer request
    auto time0 = std::chrono::high_resolution_clock::now();
    draft_model.infer();
    main_model.infer();

    size_t vocab_size = draft_model.get_tensor("logits").get_shape().back();
    OPENVINO_ASSERT(vocab_size == main_model.get_tensor("logits").get_shape().back(), "vocab size should be the same for the both models");

    // logits shape is [BATCH_SIZE, seq_len, vocab_size]
    auto logits = main_model.get_tensor("logits");
    auto data_logits = logits.data<float>() + (seq_len - 1) * vocab_size;
    int64_t out_token = std::max_element(data_logits, data_logits + vocab_size) - data_logits;

    // the first token which is fed to both draft and main netwoks on each iteration
    auto first_token = out_token;
    text_streamer.put(out_token);
    std::cout << "First token " << first_token << "\n";

    // run K infer requests on draft model and get next K prediction tokens on each iteration
    uint64_t K = 5;
    std::vector<int64_t> draft_tokens;

    // The draft model predicts tokens one by one in an auto-regressive manner, draft_input_ids length should be 1.
    draft_input_ids.set_shape({BATCH_SIZE, 1});
    draft_position_ids.set_shape({BATCH_SIZE, 1});

/* Speculative decoding works the following way. The draft model predicts the next K
   tokens one by one in an autoregressive manner, while the main model validates these
   predictions and corrects them if necessary. We go through each predicted token, and
   if a difference is detected between the draft and main model, we stop and keep the
   last token predicted by the main model. Then the draft model gets the latest main
   prediction and again tries to predict the next K tokens, repeating the cycle.

   This approach reduces the need for multiple infer requests to the main model,
   enhancing performance. For instance, in more predictable parts of text generation,
   the draft model can, in best-case scenarios, generate the next K tokens that exactly
   match the target. In tha caste the are validated in a single inference request to
   the main model (which is bigger, more accurate but slower) instead of running K
   subsequent requests.
   */
    std::map<size_t, size_t> hit_stat;
    int max_sequence_length = 128;
    while (out_token != SPECIAL_EOS_TOKEN && seq_len < max_sequence_length) {
        // infer the K next tokens with draft model
        for (int i = 0; i < K; ++i) {
            draft_input_ids.data<int64_t>()[0] = out_token;
            draft_position_ids.data<int64_t>()[0] = int64_t(seq_len + i);

            draft_is_prompt.data<bool>()[0] = false;

            draft_model_pa_manager.update_tensors(draft_input_ids);

            draft_model.infer();

            auto draft_logits = draft_model.get_tensor("logits").data<float>();
            int64_t arg_max_token = std::max_element(draft_logits, draft_logits + vocab_size) - draft_logits;
            // std::cout << "Draft model result [" << i << "]=" << arg_max_token << " for input " << out_token << "\n";
            out_token = arg_max_token;
            draft_tokens.emplace_back(arg_max_token);
        }

        // std::cout << "Finished draft execution. Generated tokens:\n";
        // for (size_t i = 0; i < draft_tokens.size(); i++) {
        //     std::cout << i << ". " << draft_tokens[i] << "\n";
        // }

        // For the main network, K tokens will be fed at once in a single infer request.
        input_ids.set_shape({BATCH_SIZE, K});
        // Set the first token for the main model to be the same as for the draft model.
        input_ids.data<int64_t>()[0] = first_token;
        // std::cout << "Set main model input tokens[0]=" << first_token << "\n";
        for (int i = 0; i < K - 1; i++) {
            // std::cout << "Set main model input tokens[" << i + 1 << "]=" << draft_tokens[i] << "\n";
            input_ids.data<int64_t>()[i + 1] = draft_tokens[i];
        }

        position_ids.set_shape({BATCH_SIZE, K});
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), seq_len);

        main_model_pa_manager.update_tensors(input_ids);
        /* Need to apply attention mask to the last N tokens, so use hack to determine it inside the kernel */
        auto ptr = is_prompt.data<bool>();
        reinterpret_cast<uint8_t*>(ptr)[0] = 2;

        main_model.infer();

        data_logits = logits.data<float>();  // [BATCH_SIZE, K, vocab_size]
        size_t disagree_idx = K - 1;

        for (size_t i = 0; i < K; i++) {
            auto start = data_logits + vocab_size * i;
            auto stop = data_logits + vocab_size * (i + 1);
            out_token = std::max_element(start, stop) - start;
            // std::cout << "Main model " << i << "th token: " << out_token << "\n";
        }
        // Iterate through the predicted tokens from the main model and compare them with draft predictions.
        // In the worst-case scenario (disagreement at the beginning), iter will increase by 1.
        // In the best-case scenario, all elements match, and K predicted tokens will be taken.
        for (size_t i = 0; i < K; i++) {
            auto start = data_logits + vocab_size * i;
            auto stop = data_logits + vocab_size * (i + 1);
            out_token = std::max_element(start, stop) - start;
            text_streamer.put(out_token);

            disagree_idx = i;
            if (out_token != draft_tokens[i] || out_token == SPECIAL_EOS_TOKEN || seq_len + disagree_idx + 1 >= max_sequence_length)
                break;

            // if (i == 2) {
            //     std::cout << "Forced exit for i=" << i << " (emulate error in draft model computation)\n";
            //     break;
            // }
        }

        // After the inference request, key/values have shape [BATCH_SIZE, seq_len + K, vocab_size].
        // Increment the sequence length by the number of matched tokens, and
        // trim the KV cache to match the new sequence length.
        seq_len += disagree_idx + 1;
        if (hit_stat.count(disagree_idx + 1) == 0)
            hit_stat[disagree_idx + 1] = 1;
        else
            hit_stat[disagree_idx + 1]++;

        draft_model_pa_manager.reduce_seq_len(K - disagree_idx - 1);
        main_model_pa_manager.reduce_seq_len(K - disagree_idx - 1);

        draft_tokens.clear();
        first_token = out_token;
    }
    text_streamer.end();

    std::cout << "Total tokens: " << seq_len << "\n";

    auto time1 = std::chrono::high_resolution_clock::now();
    auto time_res0 = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0).count();
    std::cout << "Total time: " << time_res0 << "ms\n";
    std::cout << "Hit statistic:\n";
    for (auto& entry : hit_stat)
        std::cout << entry.first << ": " << entry.second << "\n";
    exit(0);
    // Model is stateful which means that context (kv-cache) which belongs to a particular
    // text sequence is accumulated inside the model during the generation loop above.
    // This context should be reset before processing the next text sequence.
    // While it is not required to reset context in this sample as only one sequence is processed,
    // it is called for education purposes:
    draft_model.reset_state();
    main_model.reset_state();
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
