import os,pathlib
p=pathlib.Path(os.environ['TMPDIR'])/'h3-rank-project/csrc/models/minimax_h3/minimax_h3_transformer.cpp'
s=p.read_text()
n='        const auto &cache = rotary_cos_sin_cache.value();'
x='''        const auto &cache = rotary_cos_sin_cache.value();
        static thread_local bool cache_captured = false;
        const char *cache_dir = std::getenv("H3_CACHE_TRACE_DIR");
        if (cache_dir && !cache_captured && sequence_length > 10000) {
            cache_captured = true;
            const auto r = infinilm::global_state::get_tensor_model_parallel_rank_info().tp_rank;
            const auto base = std::string(cache_dir) + "/rank" + std::to_string(r);
            cache->debug(base + "_cache.bin");
            position_ids.value()->debug(base + "_pos.bin");
        }'''
assert s.count(n)==1
p.write_text(s.replace(n,x,1)); print('CACHE_TRACE_READY')
