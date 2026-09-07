"""Construct trace contexts independently of workload templates."""
from pathlib import Path
from .runtime import WorkloadConfig
from .tracing import TraceContext

def build_trace_context(config: WorkloadConfig, *, topology_role: str = "workflow") -> TraceContext:
    trace_dir = Path(config.trace_dir)
    task_dir = trace_dir / config.topology_name / config.instance_id
    trace_path = task_dir / f"{config.run_id}.jsonl"
    summary_path = task_dir / f"{config.run_id}_summary.json"
    model_outputs_path = task_dir / f"{config.run_id}_model_outputs.json"
    backend_metrics_path = task_dir / f"{config.run_id}_backend_metrics.json"
    return TraceContext(
        run_id=config.run_id,
        topology=config.topology_name,
        topology_role=topology_role,
        instance_id=config.instance_id,
        task_source=config.task_source,
        workflow_id=f"{config.topology_name}_{config.run_id}",
        trace_path=trace_path,
        summary_path=summary_path,
        random_seed=config.random_seed,
        trace_level=config.trace_level,
        export_views=config.export_trace_views,
        record_model_outputs=config.record_model_outputs,
        model_outputs_path=model_outputs_path,
        collect_backend_metrics=config.collect_backend_metrics,
        backend_metrics_path=backend_metrics_path,
        mode=config.mode,
        motif_name=config.motif_name,
        motif_instance_id=config.instance_id if config.mode == "motif" else "",
        parent_motif_id=config.parent_motif_id,
        composed_from_topologies=list(config.composed_from_topologies),
    )
