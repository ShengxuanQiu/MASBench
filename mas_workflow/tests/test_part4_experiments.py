from dataclasses import asdict
import json
from pathlib import Path

from app.freeze_corpus import freeze
from app.mixed_study import run_mixed_study, weighted_class_schedule
from app.publication import generate, preset
from app.specs import DeploymentSpec, TaskBinding


def test_weighted_schedule_is_exact_and_reproducible():
    a, counts = weighted_class_schedule(10, {"burst": 7, "context": 1.5, "dependency": 1.5}, 9)
    b, again = weighted_class_schedule(10, {"burst": 7, "context": 1.5, "dependency": 1.5}, 9)
    assert a == b and counts == again == {"burst": 7, "context": 2, "dependency": 1}
    assert dict(sorted(__import__("collections").Counter(a).items())) == counts


def _write_mixed_run(tmp_path):
    task = TaskBinding("Compare options")
    for name, exp in {
        "burst": preset("matched_parallel", task, DeploymentSpec(), requests=2),
        "dependency": preset("matched_chain", task, DeploymentSpec(), requests=2),
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(asdict(exp)))
    (tmp_path / "dep.json").write_text(json.dumps(asdict(DeploymentSpec(concurrency=2))))
    config = {
        "mode": "run",
        "classes": {"burst": {"experiment": "burst.json", "slo_sec": 10},
                    "dependency": {"experiment": "dependency.json", "slo_sec": 10}},
        "mixes": {"balanced": {"burst": 1, "dependency": 1}},
        "isolated_baselines": True,
        "deployments": {"mock": "dep.json"},
        "rates": [100], "count": 4, "repetitions": 1, "seed": 4,
        "max_inflight_workflows": 4, "slo_sec": 10,
    }
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(config))
    return path


def test_mixed_study_per_class_schedule_baseline_and_frozen_replay(tmp_path):
    path = _write_mixed_run(tmp_path)
    rows = run_mixed_study(path, tmp_path / "run")
    mixed = next(row for row in rows if row["mix"] == "balanced")
    assert mixed["exact_class_counts"] == {"burst": 2, "dependency": 2}
    assert set(mixed["per_class"]) == {"burst", "dependency"}
    assert all(row["completed"] == row["offered"] for row in mixed["per_class"].values())
    assert all(row["slowdown_p50"] is not None for row in mixed["per_class"].values())
    capacity = json.loads((tmp_path / "run" / "capacity_results.json").read_text())
    assert "balanced" in capacity["mock"]
    schedule = json.loads((Path(mixed["attempt_path"]) / "arrival_schedule.json").read_text())
    assert schedule["schedule_sha256"] == mixed["schedule_sha256"]
    assert [row["workload_class"] for row in schedule["arrivals"]].count("burst") == 2

    corpora = freeze(tmp_path / "run", tmp_path / "frozen", per_class=1)
    replay = {
        "mode": "replay",
        "classes": {name: {"trace_corpus": str(Path(corpus).resolve()), "slo_sec": 10}
                    for name, corpus in corpora.items()},
        "mixes": {"balanced": {"burst": 1, "dependency": 1}},
        "deployments": {"mock": str((tmp_path / "dep.json").resolve())},
        "rates": [100], "count": 4, "seed": 4, "max_inflight_workflows": 4,
        "slo_sec": 10, "output_length_tolerance": 0,
    }
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(replay))
    replay_rows = run_mixed_study(replay_path, tmp_path / "replay")
    assert all(check["invariants"] for check in replay_rows[0]["replay_checks"])
    # Mock does not report output-token counts, so strict length eligibility is
    # unavailable rather than silently accepted.
    assert replay_rows[0]["replay_equivalence_failures"] == 4
    assert replay_rows[0]["schedule_sha256"] == mixed["schedule_sha256"]


def test_part4_publication_manifests_expand(tmp_path):
    root = Path(__file__).parents[1]
    for name in ("4_2_pressure_signatures.json", "4_3_multiplexing.json"):
        cells = generate(root / "configs" / "part4" / name, tmp_path / name)
        assert cells
    assert len(list((root.parent / "evaluation" / "part4").glob("run_4_*.sh"))) == 5
