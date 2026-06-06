# Week3 Simulator-ready Trace Schema

The raw JSONL traces remain the source of truth. Week3 normalization exports simulator-facing tables derived only from real vLLM-backed runs.

## Workflow
`workflow_run_id`, `workflow_name`, `workflow_type`, `motif_type`, `composed_subgraphs`, `task_id`, `prompt_id`, `start_time`, `end_time`, `end_to_end_latency`, `status`, backend endpoint/model, concurrency/backend metrics.

## Graph / Node
`node_id`, `agent_id`, `agent_role`, `node_type`, `parent_node_ids`, `child_node_ids`, `dependency_type`, `branch_id`, `round_id`, `loop_iteration_id`, `is_critical_path`.

## LLM Query
`request_id`, `workflow_run_id`, `node_id`, `agent_id`, `role`, `model_name`, `submit_time`, queue/prefill/decode timestamps when available, `finish_time`, `ttft`, `tpot`, `input_tokens`, `prefill_tokens`, `output_tokens`, `decode_tokens`, prompt segment token breakdown, `prompt_hash`, `prompt_segment_hashes` / `segment_hashes`, sampling params, `status`.

Queue and prefill substage timestamps are marked `unavailable` when vLLM does not expose per-request values.

## Tool Call
`tool_call_id`, `workflow_run_id`, `node_id`, `tool_name`, start/end time, latency, input/output size, status, retry count, and whether the result is written to shared context.

## Synchronization / Barrier
`barrier_id`, `barrier_type`, participants, release time, per-node wait proxy, straggler gap, downstream nodes.

## Prefix / Cache
Actual backend metrics are sourced only from vLLM `/metrics` sidecars, including run-level GPU KV cache usage and prefix-cache counters when exposed.

Offline fields use explicit potential terminology: `potential_prefix_match_tokens`, `potential_prefix_reuse_rate`, `potential_prefix_source`, `intra_workflow_prefix_match_tokens`, `inter_workflow_prefix_match_tokens`, `shared_context_reuse_tokens`, `private_context_reuse_tokens`, `dynamic_context_new_tokens`.

`potential_prefix_source` distinguishes exact full-prompt hash matches, shared-block hash matches, and `segment_token_proxy` estimates derived from prompt template and segment token counts when old traces do not contain full token sequences.

Do not interpret potential prefix reuse as actual cache hit rate.
