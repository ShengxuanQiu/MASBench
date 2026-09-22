from app.pressure_signatures import KVModelContract, backend_resource_series, causal_waves


def test_qwen3_kv_equivalent_contract():
    contract = KVModelContract.from_model_config({
        "num_hidden_layers": 36, "num_key_value_heads": 8,
        "num_attention_heads": 32, "hidden_size": 4096,
        "torch_dtype": "bfloat16"})
    assert contract.head_dim == 128
    assert contract.bytes_per_token == 147456


def test_causal_waves_preserve_fanout_and_join():
    waves = causal_waves({"root": set(), "left": {"root"}, "right": {"root"},
                           "join": {"left", "right"}})
    assert waves == {"root": 1, "left": 2, "right": 2, "join": 3}


def test_utilization_ratio_is_not_arithmetic_intensity():
    rows = backend_resource_series([{"relative_time_sec": 1.0,
        "device_metrics": {"ai_core_utilization_percent": 60,
                           "memory_bandwidth_percent": 30},
        "cache_memory_metrics": {"kv_cache_usage_percent": 25}}])
    assert rows[0]["compute_to_hbm_utilization_ratio"] == 2
    assert rows[0]["sources"]["arithmetic_intensity_flops_per_dram_byte"] == "unavailable"
