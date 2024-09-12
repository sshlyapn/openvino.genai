// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "sequence_group.hpp"
#include "scheduler.hpp"

using namespace ov::genai;

void clear_finished_sequences(std::vector<SequenceGroup::Ptr>& requests) {
    auto new_end = std::remove_if(requests.begin(), requests.end(), [] (SequenceGroup::CPtr seq_group) -> bool {
            return seq_group->has_finished();
    });
    requests.erase(new_end, requests.end());
}

TEST(TestScheduler, general_test) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).max_num_batched_tokens = 32;
    configs.at(0).num_kv_blocks = 6;
    configs.at(0).block_size = 4;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).max_num_seqs = 5;
    configs.at(1).max_num_batched_tokens = 32;
    configs.at(1).num_kv_blocks = 6;
    configs.at(1).block_size = 4;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).max_num_seqs = 5;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> tokens = {0,1,2,3,4,5,6,7};
        SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
        auto idx0 = (*sequence_group1)[0]->get_id();
        SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
        auto idx1 = (*sequence_group2)[0]->get_id();
        SequenceGroup::Ptr sequence_group3 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
        auto idx2 = (*sequence_group3)[0]->get_id();
        std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2, sequence_group3};
                                                                        
        
        // schedule 3 sequence groups that use 6 kv blocks 
        Scheduler scheduler = Scheduler(scheduler_config);
        auto out1 = scheduler.schedule(requests);

        std::vector<uint64_t> ref_ids = {0, 1, 2};
        EXPECT_EQ(out1.m_scheduled_sequence_groups_ids, ref_ids);
        EXPECT_EQ(out1.m_block_tables[idx0].size(), 2);
        EXPECT_EQ(out1.m_block_tables[idx1].size(), 2);
        EXPECT_EQ(out1.m_block_tables[idx2].size(), 2);
        // tokens.size() * 2 tokens should be scheduled on prompt phase, corresponding to first three sequences 
        EXPECT_EQ(out1.m_total_num_scheduled_tokens, tokens.size() * 3);
        EXPECT_EQ(out1.is_prompt, !scheduler_config.dynamic_split_fuse);

        for (auto seq: requests) {
            std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
            // prompt phase
            seq->finish_iteration();
        }

        // at this point we scheduled all available kv blocks

        // sequence_group3 should be evicted
        auto out3 = scheduler.schedule(requests);

        for (auto seq: requests) {
            std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
            // generate phase, append a token to each sequence
            running_sequences[0]->append_token(16, 0.9);
            seq->finish_iteration();
        }

        std::vector<uint64_t> ref_ids2 = {0, 1};
        EXPECT_EQ(out3.m_scheduled_sequence_groups_ids, ref_ids2);
        EXPECT_EQ(out3.m_block_tables[idx0].size(), 3);
        EXPECT_EQ(out3.m_block_tables[idx1].size(), 3);
        // 2 tokens should be scheduled on generate phase for "0" and "1" sequence, "2" sequence should be preempted
        EXPECT_EQ(out3.m_total_num_scheduled_tokens, 2); 
        EXPECT_FALSE(out3.is_prompt);

        // check that scheduler has no block table for sequence_group3
        EXPECT_FALSE(scheduler.has_block_table(idx2));

        // finish first sequence
        requests[0]->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
        scheduler.free_sequence(idx0);
        clear_finished_sequences(requests);
        // KV blocks 0,1,5 are free now


        auto out4 = scheduler.schedule(requests);

        // check that sequence_group3 is fully scehuled 
        EXPECT_EQ(out4.m_block_tables[idx2].size(), 2);
        EXPECT_FALSE(out4.m_block_tables[idx2][0]->is_free());
        EXPECT_EQ(out4.m_block_tables[idx2][0]->get_index(), 0);
        EXPECT_FALSE(out4.m_block_tables[idx2][1]->is_free());
        EXPECT_EQ(out4.m_block_tables[idx2][1]->get_index(), 1);

        // requests1[1] should be fully scheduled plus 1 slot for requests[0] for generate phase
        EXPECT_EQ(out4.m_total_num_scheduled_tokens, requests[1]->get_context_len() + 1);
        EXPECT_EQ(out4.is_prompt, false);
    }

}

TEST(TestScheduler, test_append_slots_considers_all_sequences) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).max_num_batched_tokens = 32;
    configs.at(0).num_kv_blocks = 5;
    configs.at(0).block_size = 4;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).max_num_seqs = 5;
    configs.at(1).max_num_batched_tokens = 32;
    configs.at(1).num_kv_blocks = 5;
    configs.at(1).block_size = 4;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).max_num_seqs = 5;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> tokens = {0,1,2,3,4,5,6,7};
        SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
        auto idx0 = (*sequence_group1)[0]->get_id();
        SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
        auto idx1 = (*sequence_group2)[0]->get_id();
        std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};
        
        Scheduler scheduler = Scheduler(scheduler_config);
        auto out1 = scheduler.schedule(requests);

        std::vector<uint64_t> ref_ids = {0, 1};
        EXPECT_EQ(out1.m_scheduled_sequence_groups_ids, ref_ids);
        EXPECT_EQ(out1.m_block_tables[idx0].size(), 2);
        EXPECT_EQ(out1.m_block_tables[idx1].size(), 2);
        EXPECT_FALSE(out1.m_block_tables[idx0][0]->is_free());
        EXPECT_EQ(out1.m_block_tables[idx0][0]->get_index(), 0);
        EXPECT_FALSE(out1.m_block_tables[idx0][1]->is_free());
        EXPECT_EQ(out1.m_block_tables[idx0][1]->get_index(), 1);
        EXPECT_FALSE(out1.m_block_tables[idx1][0]->is_free());
        EXPECT_EQ(out1.m_block_tables[idx1][0]->get_index(), 2);
        EXPECT_FALSE(out1.m_block_tables[idx1][1]->is_free());
        EXPECT_EQ(out1.m_block_tables[idx1][1]->get_index(), 3);
        EXPECT_EQ(out1.m_total_num_scheduled_tokens, tokens.size() * 2);
        EXPECT_EQ(out1.is_prompt, !scheduler_config.dynamic_split_fuse);
        for (auto seq: requests) {
            std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
            // prompt phase
            seq->finish_iteration();
        }

        // at this point we used 4/5 KV blocks. Both sequences require new KV block, but we have space for only one.
        auto out2 = scheduler.schedule(requests); 

        // 1-st sequence now should use 3 kv-blocks
        EXPECT_EQ(out2.m_block_tables[idx0].size(), 3);
        EXPECT_FALSE(out2.m_block_tables[idx0][0]->is_free());
        EXPECT_EQ(out2.m_block_tables[idx0][0]->get_index(), 0);
        EXPECT_FALSE(out2.m_block_tables[idx0][1]->is_free());
        EXPECT_EQ(out2.m_block_tables[idx0][1]->get_index(), 1);
        EXPECT_FALSE(out2.m_block_tables[idx0][2]->is_free());
        EXPECT_EQ(out2.m_block_tables[idx0][2]->get_index(), 4);

        // 1 token was scheduled for generate phase
        EXPECT_EQ(out2.m_total_num_scheduled_tokens, 1); 

        EXPECT_FALSE(out2.is_prompt); 
    }
}


TEST(TestScheduler, test_partial_preemption) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).max_num_batched_tokens = 32;
    configs.at(0).num_kv_blocks = 6;
    configs.at(0).block_size = 4;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).max_num_seqs = 5;
    configs.at(1).max_num_batched_tokens = 32;
    configs.at(1).num_kv_blocks = 6;
    configs.at(1).block_size = 4;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).max_num_seqs = 5;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> tokens1 = {0,1,2,3,4,5,6,7,8,9,10};
        SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens1.size()}, tokens1.data()),
                                                                                ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
        std::vector<uint64_t> tokens2 = {0,1,2,3,4,5,6,7};
        auto idx0 = (*sequence_group1)[0]->get_id();
        SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens2.size()}, tokens2.data()),
                                                                                ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
        auto idx1 = (*sequence_group2)[0]->get_id();
        std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};
                                                                        
        
        // schedule 2 sequence groups that use 5 kv blocks
        Scheduler scheduler = Scheduler(scheduler_config);
        auto out0 = scheduler.schedule(requests);

        for (auto seq: requests) {
            std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
            // prompt phase
            seq->finish_iteration();
        }


        // schedule generate, all 6 kv blocks are used.
        auto out1 = scheduler.schedule(requests);

        for (auto seq: requests) {
            std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
            // generate phase
            running_sequences[0]->append_token(16, 0.9);
            seq->finish_iteration();
        }

        // sequence_group2 should be partially preempted
        auto out2 = scheduler.schedule(requests);
        
        std::vector<uint64_t> ref_ids = {0};
        EXPECT_EQ(out2.m_scheduled_sequence_groups_ids, ref_ids);
        auto block_table1 = scheduler.get_block_table(*(*sequence_group1)[0]);
        auto block_table2 = scheduler.get_block_table(*(*sequence_group2)[0]);
        EXPECT_EQ(block_table1.size(), 4);
        EXPECT_EQ(block_table1[0]->get_index(), 0);
        EXPECT_EQ(block_table1[1]->get_index(), 1);
        EXPECT_EQ(block_table1[2]->get_index(), 2);
        EXPECT_EQ(block_table1[3]->get_index(), 5);
        EXPECT_EQ(block_table2.size(), 2);
        EXPECT_EQ(block_table2[0]->get_index(), 3);
        EXPECT_EQ(block_table2[1]->get_index(), 4);

        EXPECT_EQ(out2.m_total_num_scheduled_tokens, 1); 
        EXPECT_EQ(out2.m_block_tables[idx0][0]->get_index(), 0);
        EXPECT_EQ(out2.m_block_tables[idx0][1]->get_index(), 1);
        EXPECT_EQ(out2.m_block_tables[idx0][2]->get_index(), 2);
        EXPECT_EQ(out2.m_block_tables[idx0][3]->get_index(), 5);

        // finish first sequence
        requests[0]->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
        scheduler.free_sequence(idx0);
        clear_finished_sequences(requests);
        // KV blocks 0,1,2,5 are free now

        // sequence_group2 should be scheduled
        auto out3 = scheduler.schedule(requests);

        // last token should be recomputed
        EXPECT_EQ(out3.m_total_num_scheduled_tokens, 1); 
        EXPECT_EQ(out3.m_block_tables[idx1][0]->get_index(), 3);
        EXPECT_EQ(out3.m_block_tables[idx1][1]->get_index(), 4);
        EXPECT_EQ(out3.m_block_tables[idx1][2]->get_index(), 0);

        block_table2 = scheduler.get_block_table(*(*sequence_group2)[0]);
        EXPECT_EQ(block_table2.size(), 3);
        EXPECT_EQ(block_table2[0]->get_index(), 3);
        EXPECT_EQ(block_table2[1]->get_index(), 4);
        EXPECT_EQ(block_table2[2]->get_index(), 0);

        EXPECT_FALSE(scheduler.has_block_table(idx0));
    }
}


TEST(TestScheduler, test_partial_preemption_beam_search) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).num_kv_blocks = 10;
    configs.at(0).block_size = 4;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(1).num_kv_blocks = 10;
    configs.at(1).block_size = 4;
    configs.at(1).dynamic_split_fuse = true;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> tokens = {0,1,2,3};
        int64_t token = 4;

        // create beam search group
        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                ov::genai::beam_search(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
        sequence_group->set_sequence_group_ptr(sequence_group);
        std::vector<SequenceGroup::Ptr> requests = {sequence_group};

        Scheduler scheduler = Scheduler(scheduler_config);
        auto out = scheduler.schedule(requests);
        for (auto sequence: sequence_group->get_not_finished_sequences()) {
            sequence->append_token(token, 0.7);
        }
        sequence_group->finish_iteration();

        // make 2 forked sequence
        auto sequence_to_fork = sequence_group->get_running_sequences()[0];    
        for (size_t i = 0; i < 2; ++i) {
            const auto forked_sequence = sequence_group->fork_sequence(sequence_to_fork);
            scheduler.fork_sequence(sequence_to_fork->get_id(), forked_sequence->get_id());
        }
        size_t num_scheduled_tokens = 4;

        // generate 4 tokens
        for (size_t i = 0; i < num_scheduled_tokens; i++) {
            scheduler.schedule(requests);
            for (auto sequence: sequence_group->get_not_finished_sequences()) {
                token += 3;
                sequence->append_token(token, 0.5);
            }
            sequence_group->finish_iteration();
        }
        // currently sequence occupies 4 blocks (1 shared, 3 not shared) 

        // make another 2 forked sequence
        for (size_t i = 0; i < 2; ++i) {
            const auto forked_sequence = sequence_group->fork_sequence(sequence_to_fork);
            scheduler.fork_sequence(sequence_to_fork->get_id(), forked_sequence->get_id());
        }

        // generate 4 tokens
        for (size_t i = 0; i < num_scheduled_tokens; i++) {
            scheduler.schedule(requests);
            for (auto sequence: sequence_group->get_not_finished_sequences()) {
                token += 3;
                sequence->append_token(token, 0.5);
            }
            sequence_group->finish_iteration();
        }
        // currently sequence occupies 9 blocks (4 blocks previously created + 5 blocks for each sequence)

        // create group, which requires 1 block
        SequenceGroup::Ptr sequence_group_greedy = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
        sequence_group_greedy->set_sequence_group_ptr(sequence_group_greedy);

        // set greedy group at the beginning of list to make it higher priority
        std::vector<SequenceGroup::Ptr> new_requests = {sequence_group_greedy, sequence_group};

        // process prompt of greedy group, at this point all blocks are used
        scheduler.schedule(new_requests);
        sequence_group_greedy->get_sequences()[0]->append_token(token, 0.8);
        sequence_group_greedy->finish_iteration();

        EXPECT_EQ(sequence_group->get_num_processed_tokens(), 12);
        EXPECT_EQ(sequence_group->get_context_len(), 12);
        
        // beam search group should be partially preempted and 5 blocks should be released 
        out = scheduler.schedule(new_requests);
        sequence_group_greedy->get_sequences()[0]->append_token(token, 0.5);
        sequence_group_greedy->finish_iteration();

        EXPECT_EQ(sequence_group->get_num_processed_tokens(), 8);
        auto seqs = sequence_group->get_sequences();
        EXPECT_EQ(scheduler.get_block_table(*seqs[0]).size(), 2);
        EXPECT_EQ(scheduler.get_block_table(*seqs[1]).size(), 2);
        EXPECT_EQ(scheduler.get_block_table(*seqs[2]).size(), 2);
        EXPECT_EQ(scheduler.get_block_table(*seqs[3]).size(), 2);
        EXPECT_EQ(scheduler.get_block_table(*seqs[4]).size(), 2);
        
        // append another 20 tokens to greedy group, this should result in usage of all free blocks and 
        // another partial preemption of beam search group
        for (size_t i = 0; i < 20; i++) {
            out = scheduler.schedule(new_requests);
            sequence_group_greedy->get_sequences()[0]->append_token(token, 0.5);
            sequence_group_greedy->finish_iteration();
        }

        EXPECT_EQ(sequence_group->get_num_processed_tokens(), 4);
        seqs = sequence_group->get_sequences();
        EXPECT_EQ(scheduler.get_block_table(*seqs[0]).size(), 1);
        EXPECT_EQ(scheduler.get_block_table(*seqs[1]).size(), 1);
        EXPECT_EQ(scheduler.get_block_table(*seqs[2]).size(), 1);
        EXPECT_EQ(scheduler.get_block_table(*seqs[3]).size(), 1);
        EXPECT_EQ(scheduler.get_block_table(*seqs[4]).size(), 1);
    }
}

TEST(TestScheduler, test_partially_preempted_prompt) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).max_num_batched_tokens = 32;
    configs.at(0).num_kv_blocks = 6;
    configs.at(0).block_size = 4;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).max_num_seqs = 5;
    configs.at(1).max_num_batched_tokens = 32;
    configs.at(1).num_kv_blocks = 6;
    configs.at(1).block_size = 4;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).max_num_seqs = 5;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> tokens = {0,1,2,3,4,5,6,7,8,9,10,11};
        SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
        auto idx0 = (*sequence_group1)[0]->get_id();
        SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
        auto idx1 = (*sequence_group2)[0]->get_id();
        std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};
                                                                        
        
        // schedule 2 sequence groups that use all available 2*3 kv blocks, we used all available kv-blocks.
        Scheduler scheduler = Scheduler(scheduler_config);
        auto out1 = scheduler.schedule(requests);

        for (auto seq: requests) {
            std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
            // prompt phase
            seq->finish_iteration();
        }

        // sequence_group2 should be fully preempted
        auto out2 = scheduler.schedule(requests);
        
        // check that sequence_group1 has one more allocated block
        auto block_table1 = scheduler.get_block_table(*(*sequence_group1)[0]);
        EXPECT_EQ(block_table1.size(), 4);
        EXPECT_EQ(block_table1[0]->get_index(), 0);
        EXPECT_EQ(block_table1[1]->get_index(), 1);
        EXPECT_EQ(block_table1[2]->get_index(), 2);
        EXPECT_EQ(block_table1[3]->get_index(), 5);
        EXPECT_EQ(out2.m_block_tables[idx0].size(), 4);
        EXPECT_EQ(out2.m_block_tables[idx0][0]->get_index(), 0);
        EXPECT_EQ(out2.m_block_tables[idx0][1]->get_index(), 1);
        EXPECT_EQ(out2.m_block_tables[idx0][2]->get_index(), 2);
        EXPECT_EQ(out2.m_block_tables[idx0][3]->get_index(), 5);

        std::vector<uint64_t> ref_ids = {0};
        EXPECT_EQ(out2.m_scheduled_sequence_groups_ids, ref_ids);
        EXPECT_EQ(out2.m_total_num_scheduled_tokens, 1); 

        if (scheduler_config.dynamic_split_fuse) {
            // for dynamic_split_fuse sequence_group2 is preemted partially, part of prompt is left
            EXPECT_TRUE(scheduler.has_block_table(idx1));
            auto block_table2 = scheduler.get_block_table(*(*sequence_group2)[0]);
            EXPECT_EQ(block_table2.size(), 2); // full prompt requires 3 blocks, 2 are left in scheduler

        } else {
            // for vllm case sequence_group2 is fully preempted
            EXPECT_FALSE(scheduler.has_block_table(idx1));
        }
        
        for (auto seq: requests) {
            std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
            seq->finish_iteration();
        }
        
        // finish first sequence
        requests[0]->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
        scheduler.free_sequence(idx0);
        clear_finished_sequences(requests);
        // KV blocks 0,1,2,5 are free now

        // sequence_group2 should be scheduled
        auto out3 = scheduler.schedule(requests);

        if (scheduler_config.dynamic_split_fuse) {
            // remaining part of prompt should be scheduled
            EXPECT_EQ(out3.m_total_num_scheduled_tokens, 4); 
        }
        else {
            // prompt should be fully scheduled
            EXPECT_EQ(out3.m_total_num_scheduled_tokens, 12); 
        }

        EXPECT_EQ(out3.m_block_tables[idx1][0]->get_index(), 3);
        EXPECT_EQ(out3.m_block_tables[idx1][1]->get_index(), 4);
        EXPECT_EQ(out3.m_block_tables[idx1][2]->get_index(), 0);

        auto block_table2 = scheduler.get_block_table(*(*sequence_group2)[0]);
        EXPECT_EQ(block_table2.size(), 3);
        EXPECT_EQ(block_table2[0]->get_index(), 3);
        EXPECT_EQ(block_table2[1]->get_index(), 4);
        EXPECT_EQ(block_table2[2]->get_index(), 0);

        EXPECT_FALSE(scheduler.has_block_table(idx0));
    }
}

TEST(TestScheduler, prefix_caching_test) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).max_num_batched_tokens = 32;
    configs.at(0).num_kv_blocks = 100;
    configs.at(0).block_size = 4;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).max_num_seqs = 5;
    configs.at(0).enable_prefix_caching = true;
    configs.at(1).max_num_batched_tokens = 32;
    configs.at(1).num_kv_blocks = 100;
    configs.at(1).block_size = 4;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).max_num_seqs = 5;
    configs.at(1).enable_prefix_caching = true;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> prompt_tokens = {0,1,2,3,4,5,6,7};
        std::vector<uint64_t> histrory_tokens = {};
        // schedule prompt
        Scheduler scheduler = Scheduler(scheduler_config);

        size_t chat_iterations = 10;

        for (size_t chat_iteration = 0; chat_iteration < chat_iterations; chat_iteration++) {
            std::vector<uint64_t> tokens = histrory_tokens;
            tokens.insert(tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
            SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                    ov::genai::greedy(), scheduler_config.block_size, 
                                                                                    scheduler_config.enable_prefix_caching);
            sequence_group->set_sequence_group_ptr(sequence_group);
            scheduler.restore_cached_blocks(sequence_group);
            std::vector<SequenceGroup::Ptr> requests = {sequence_group};

            auto out1 = scheduler.schedule(requests);
            if (chat_iteration == 0)
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, prompt_tokens.size());
            else 
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, prompt_tokens.size() + 1);
            for (auto seq: requests) {
                std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
                running_sequences[0]->append_token(23, 0.7);
                seq->finish_iteration();
            }

            // schedule generate
            size_t num_generate_tokens = 10;
            for (size_t i = 0; i < num_generate_tokens; i++) {
                auto out2 = scheduler.schedule(requests);
                EXPECT_EQ(out2.m_total_num_scheduled_tokens, 1);
                for (auto seq: requests) {
                    std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
                    running_sequences[0]->append_token(16, 0.9);
                    seq->finish_iteration();
                }
            }

            // finish sequence
            auto sequence = requests[0]->get_running_sequences()[0];
            sequence->set_status(SequenceStatus::FINISHED);
            auto idx0 = sequence->get_id();
            scheduler.free_sequence(idx0);
            auto generated_ids = sequence->get_generated_ids();

            histrory_tokens.insert(histrory_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
            histrory_tokens.insert(histrory_tokens.end(), generated_ids.begin(), generated_ids.end());
        }
    }

}

TEST(TestScheduler, prefix_caching_test_two_identical_sequences) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).num_kv_blocks = 100;
    configs.at(0).block_size = 4;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).enable_prefix_caching = true;
    configs.at(1).num_kv_blocks = 100;
    configs.at(1).block_size = 4;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).enable_prefix_caching = true;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> prompt_tokens = {0,1,2,3,4,5,6,7};
        std::vector<uint64_t> histrory_tokens = {};
        // schedule prompt
        Scheduler scheduler = Scheduler(scheduler_config);

        size_t chat_iterations = 10;

        for (size_t chat_iteration = 0; chat_iteration < chat_iterations; chat_iteration++) {
            std::vector<uint64_t> tokens = histrory_tokens;
            tokens.insert(tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
            SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                    ov::genai::greedy(), scheduler_config.block_size, 
                                                                                    scheduler_config.enable_prefix_caching);

            SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                    ov::genai::greedy(), scheduler_config.block_size, 
                                                                                    scheduler_config.enable_prefix_caching);
            sequence_group1->set_sequence_group_ptr(sequence_group1);
            sequence_group2->set_sequence_group_ptr(sequence_group2);
            std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};
            // restore cached blocks
            for (auto request: requests) {
                scheduler.restore_cached_blocks(request);
            }

            // schedule prompt
            auto out1 = scheduler.schedule(requests);
            if (chat_iteration == 0)
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, prompt_tokens.size() * 2);
            else 
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, (prompt_tokens.size() + 1) * 2);
            for (auto seq: requests) {
                std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
                running_sequences[0]->append_token(23, 0.7);
                seq->finish_iteration();
            }

            // schedule generate
            size_t num_generate_tokens = 10;
            for (size_t i = 0; i < num_generate_tokens; i++) {
                auto out2 = scheduler.schedule(requests);
                EXPECT_EQ(out2.m_total_num_scheduled_tokens, 2);
                for (auto request: requests) {
                    std::vector<Sequence::Ptr> running_sequences = request->get_running_sequences();
                    running_sequences[0]->append_token(16, 0.9);
                    request->finish_iteration();
                }
            }

            for (auto request: requests) {
                // finish sequences
                auto sequence = request->get_running_sequences()[0];
                sequence->set_status(SequenceStatus::FINISHED);
                auto idx0 = sequence->get_id();
                scheduler.free_sequence(idx0);
            }
            auto generated_ids = requests[0]->get_sequences()[0]->get_generated_ids();
            
            histrory_tokens.insert(histrory_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
            histrory_tokens.insert(histrory_tokens.end(), generated_ids.begin(), generated_ids.end());
        }
    }

}


TEST(TestScheduler, prefix_caching_with_max_new_tokens_equal_1) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).num_kv_blocks = 10;
    configs.at(0).block_size = 32;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).enable_prefix_caching = true;
    configs.at(1).num_kv_blocks = 10;
    configs.at(1).block_size = 32;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).enable_prefix_caching = true;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> prompt_tokens = {0,1,2,3,4,5,6,7};
        // schedule prompt
        Scheduler scheduler = Scheduler(scheduler_config);

        size_t chat_iterations = 2;

        for (size_t chat_iteration = 0; chat_iteration < chat_iterations; chat_iteration++) {
            SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
                                                                                    ov::genai::greedy(), scheduler_config.block_size, 
                                                                                    scheduler_config.enable_prefix_caching);

            sequence_group->set_sequence_group_ptr(sequence_group);
            std::vector<SequenceGroup::Ptr> requests = {sequence_group};
            // restore cached blocks
            for (auto request: requests) {
                scheduler.restore_cached_blocks(request);
            }

            // schedule prompt
            auto out1 = scheduler.schedule(requests);
            if (chat_iteration == 0)
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, prompt_tokens.size());
            else 
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, 1);
            for (auto seq: requests) {
                std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
                running_sequences[0]->append_token(23, 0.7);
                seq->finish_iteration();
            }

            // In case max_new_tokens == 1 no generate phase happens

            for (auto request: requests) {
                // finish sequences
                auto sequence = request->get_running_sequences()[0];
                sequence->set_status(SequenceStatus::FINISHED);
                auto idx0 = sequence->get_id();
                scheduler.free_sequence(idx0);
            }
        }
    }

}

TEST(TestScheduler, test_partially_preempted_prompt_not_allowed) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 6;
    scheduler_config.block_size = 4;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 5;

    std::vector<uint64_t> tokens = {0,1,2,3,4,5,6,7,8,9,10,11};
    SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
    auto idx0 = (*sequence_group1)[0]->get_id();
    SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
    auto idx1 = (*sequence_group2)[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};


    // schedule 2 sequence groups that use all available 2*3 kv blocks, we used all available kv-blocks.
    const bool can_use_partial_preemption = false;
    Scheduler scheduler = Scheduler(scheduler_config, can_use_partial_preemption);
    auto out1 = scheduler.schedule(requests);

    for (auto req : requests)
        req->finish_iteration();

    // sequence_group2 should be fully preempted
    auto out2 = scheduler.schedule(requests);

    // check that sequence_group1 has one more allocated block
    auto block_table1 = scheduler.get_block_table(*(*sequence_group1)[0]);
    ASSERT_EQ(block_table1.size(), 4);
    ASSERT_EQ(block_table1[0]->get_index(), 0);
    ASSERT_EQ(block_table1[1]->get_index(), 1);
    ASSERT_EQ(block_table1[2]->get_index(), 2);
    ASSERT_EQ(block_table1[3]->get_index(), 3);
    ASSERT_EQ(out2.m_block_tables[idx0].size(), 4);
    ASSERT_EQ(out2.m_block_tables[idx0][0]->get_index(), 0);
    ASSERT_EQ(out2.m_block_tables[idx0][1]->get_index(), 1);
    ASSERT_EQ(out2.m_block_tables[idx0][2]->get_index(), 2);
    ASSERT_EQ(out2.m_block_tables[idx0][3]->get_index(), 3);

    std::vector<uint64_t> ref_ids = {0};
    ASSERT_EQ(out2.m_scheduled_sequence_groups_ids, ref_ids);
    ASSERT_EQ(out2.m_total_num_scheduled_tokens, 1);

    // for vllm case sequence_group2 is fully preempted
    EXPECT_FALSE(scheduler.has_block_table(idx1));

    for (auto req : requests)
        req->finish_iteration();

    // finish first sequence
    requests[0]->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
    scheduler.free_sequence(idx0);
    clear_finished_sequences(requests);

    // sequence_group2 should be scheduled
    auto out3 = scheduler.schedule(requests);

    // prompt should be fully scheduled
    ASSERT_EQ(out3.m_total_num_scheduled_tokens, 12);

    ASSERT_EQ(out3.m_block_tables[idx1][0]->get_index(), 4);
    ASSERT_EQ(out3.m_block_tables[idx1][1]->get_index(), 5);
    ASSERT_EQ(out3.m_block_tables[idx1][2]->get_index(), 0);

    auto block_table2 = scheduler.get_block_table(*(*sequence_group2)[0]);
    ASSERT_EQ(block_table2.size(), 3);
    ASSERT_EQ(block_table2[0]->get_index(), 4);
    ASSERT_EQ(block_table2[1]->get_index(), 5);
    ASSERT_EQ(block_table2[2]->get_index(), 0);

    EXPECT_FALSE(scheduler.has_block_table(idx0));
}

TEST(TestScheduler, test_partially_preempted_prompt_not_allowed2) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 6;
    scheduler_config.block_size = 4;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 5;

    std::vector<uint64_t> tokens = {0,1,2,3,4,5,6,7,8,9};
    SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
    auto idx0 = (*sequence_group1)[0]->get_id();
    SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            ov::genai::greedy(), scheduler_config.block_size, scheduler_config.enable_prefix_caching);
    auto idx1 = (*sequence_group2)[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};

    // schedule 2 sequence groups that use all available 2*3 kv blocks, we used all available kv-blocks.
    const bool can_use_partial_preemption = false;
    Scheduler scheduler = Scheduler(scheduler_config, can_use_partial_preemption);
    scheduler.schedule(requests);
    for (auto req: requests)
        req->finish_iteration();

    scheduler.schedule(requests);
    for (auto req: requests)
        req->finish_iteration();

    scheduler.schedule(requests);
    for (auto req: requests)
        req->finish_iteration();

    // sequence_group2 should be fully preempted
    scheduler.schedule(requests);
    for (auto req: requests)
        req->finish_iteration();

    auto out2 = scheduler.schedule(requests);

    // check that sequence_group1 has one more allocated block
    auto block_table1 = scheduler.get_block_table(*(*sequence_group1)[0]);
    ASSERT_EQ(block_table1.size(), 4);
    ASSERT_EQ(block_table1[0]->get_index(), 0);
    ASSERT_EQ(block_table1[1]->get_index(), 1);
    ASSERT_EQ(block_table1[2]->get_index(), 2);
    ASSERT_EQ(block_table1[3]->get_index(), 3);
    ASSERT_EQ(out2.m_block_tables[idx0].size(), 4);
    ASSERT_EQ(out2.m_block_tables[idx0][0]->get_index(), 0);
    ASSERT_EQ(out2.m_block_tables[idx0][1]->get_index(), 1);
    ASSERT_EQ(out2.m_block_tables[idx0][2]->get_index(), 2);
    ASSERT_EQ(out2.m_block_tables[idx0][3]->get_index(), 3);

    std::vector<uint64_t> ref_ids = {0};
    ASSERT_EQ(out2.m_scheduled_sequence_groups_ids, ref_ids);
    ASSERT_EQ(out2.m_total_num_scheduled_tokens, 1);

    // for vllm case sequence_group2 is fully preempted
    EXPECT_FALSE(scheduler.has_block_table(idx1));

    for (auto req: requests)
        req->finish_iteration();

    // finish first sequence
    requests[0]->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
    scheduler.free_sequence(idx0);
    clear_finished_sequences(requests);

    // sequence_group2 should be scheduled
    auto out3 = scheduler.schedule(requests);

    // prompt should be fully scheduled + generated tokens concatenated to prompt (10 + 2)
    ASSERT_EQ(out3.m_total_num_scheduled_tokens, 12);

    ASSERT_EQ(out3.m_block_tables[idx1][0]->get_index(), 4);
    ASSERT_EQ(out3.m_block_tables[idx1][1]->get_index(), 5);
    ASSERT_EQ(out3.m_block_tables[idx1][2]->get_index(), 0);

    auto block_table2 = scheduler.get_block_table(*(*sequence_group2)[0]);
    ASSERT_EQ(block_table2.size(), 3);
    ASSERT_EQ(block_table2[0]->get_index(), 4);
    ASSERT_EQ(block_table2[1]->get_index(), 5);
    ASSERT_EQ(block_table2[2]->get_index(), 0);

    EXPECT_FALSE(scheduler.has_block_table(idx0));
}
