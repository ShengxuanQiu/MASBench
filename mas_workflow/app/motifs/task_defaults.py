"""Default task prompts; no backend or runtime state belongs here."""
from .contracts import RoleSlot

FAMILY_SLOTS = {
    "dispatch_execute": (
        RoleSlot("dispatcher", "Produce concrete instructions for the executor to address the task."),
        RoleSlot("executor", "Execute the supplied instructions using the task and inputs."),
    ),
    "parallel_aggregate": (
        RoleSlot("worker", "Independently address the task from your assigned perspective."),
        RoleSlot("collector", "Combine the worker results into an answer to the task."),
    ),
    "evaluate_refine": (
        RoleSlot("producer", "Produce or revise the candidate using the supplied evaluation feedback."),
        RoleSlot("evaluator", "Evaluate the candidate against the task criteria."),
    ),
    "peer_deliberation": (
        RoleSlot("peer", "Address the task, then update your result using the delivered peer messages."),
    ),
}
