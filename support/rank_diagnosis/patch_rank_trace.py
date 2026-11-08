import os
from pathlib import Path
p=Path(os.environ['TMPDIR'])/'h3-rank-project/csrc/models/minimax_h3/minimax_h3_transformer.cpp'; s=p.read_text()
s=s.replace('#include <stdexcept>', '#include <cstdlib>\n#include <stdexcept>',1)
needle='''    const size_t sequence_length = hidden_states->size(1);

    auto hidden_states_mutable = hidden_states;'''
extra='''    const size_t sequence_length = hidden_states->size(1);
    static thread_local bool rank_traced = false;
    const char *rank_dir = std::getenv("H3_RANK_TRACE_DIR");
    const bool rank_capture = rank_dir && !rank_traced && sequence_length > 10000;
    const auto h3_rank = infinilm::global_state::get_tensor_model_parallel_rank_info().tp_rank;
    auto trace_rank = [&](const char *name, const infinicore::Tensor &value, size_t seq_dim) {
        if (!rank_capture) return;
        auto sample = value->narrow({{seq_dim, 0, 16}});
        sample->debug(std::string(rank_dir) + "/native_rank" + std::to_string(h3_rank) + "_" + name + ".bin");
    };
    if (rank_capture) rank_traced = true;
    trace_rank("input", hidden_states, 1);

    auto hidden_states_mutable = hidden_states;'''
assert s.count(needle)==1
s=s.replace(needle,extra,1)
needle='''    value = value->view({batch_size, sequence_length, num_heads_, head_dim_});

    query = norm_q_->forward(query);
    key = norm_k_->forward(key);'''
extra='''    value = value->view({batch_size, sequence_length, num_heads_, head_dim_});
    trace_rank("q_raw", query, 0);
    trace_rank("k_raw", key, 0);
    trace_rank("v_raw", value, 1);
    query = norm_q_->forward(query);
    key = norm_k_->forward(key);
    trace_rank("q_norm", query, 0);
    trace_rank("k_norm", key, 0);'''
assert s.count(needle)==1
s=s.replace(needle,extra,1)
needle='''    auto output = infinicore::op::mha(query, key, value, std::nullopt, scale_, false);
    output = output->view({batch_size, sequence_length, inner_dim_});'''
extra='''    auto output = infinicore::op::mha(query, key, value, std::nullopt, scale_, false);
    trace_rank("mha_out", output, 1);
    output = output->view({batch_size, sequence_length, inner_dim_});'''
assert s.count(needle)==1
p.write_text(s.replace(needle,extra,1)); print('All-rank input/QKV instrumentation prepared')
