"""Domain bindings of four families, not additional motif categories.

Historical implementations remain available through legacy_registry for exact
reproduction. These presets intentionally execute the new family semantics.
"""
from copy import deepcopy

PRESETS = {
    "planner_executor": {"family": "dispatch_execute", "width": 1},
    "router_handoff": {"family": "dispatch_execute", "dispatch": "route"},
    "evidence_collection": {"family": "parallel_aggregate", "roles": {
        "worker": {"instructions": "Collect evidence relevant to the task from your assigned perspective.", "tools": ["search"]},
        "collector": {"instructions": "Synthesize the evidence and distinguish evidence from inference."}}},
    "researcher_synthesizer": {"family": "parallel_aggregate", "roles": {
        "worker": {"instructions": "Independently research the task and explain supporting evidence."}}},
    "multi_coder_branch": {"family": "parallel_aggregate", "aggregation": "judge", "roles": {
        "worker": {"instructions": "Propose a candidate code change for the task."},
        "collector": {"instructions": "Select the code candidate best satisfying the task constraints."}}},
    "tool_specialist_team": {"family": "parallel_aggregate", "roles": {
        "worker": {"instructions": "Use the supplied tool evidence to address the task.", "tools": ["search"]}}},
    "generator_verifier": {"family": "evaluate_refine"},
    "coder_reviewer": {"family": "evaluate_refine", "roles": {
        "producer": {"instructions": "Produce or revise a code change addressing the task."},
        "evaluator": {"instructions": "Review the proposed code against the task and give actionable feedback."}}},
    "retry_debug_loop": {"family": "evaluate_refine", "roles": {
        "producer": {"instructions": "Produce or debug a solution using the supplied feedback."},
        "evaluator": {"instructions": "Inspect the solution for failures. This is an LLM assessment, not a test execution."}}},
    "debate_reviewer": {"family": "peer_deliberation", "roles": {
        "peer": {"instructions": "Review the task independently, then revise your assessment using incoming peer reviews."}}},
    "all_gather_round": {"family": "peer_deliberation", "connectivity": "all_to_all"},
}


def preset(name):
    return deepcopy(PRESETS[name])
