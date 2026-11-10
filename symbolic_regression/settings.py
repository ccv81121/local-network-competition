"""Search settings.

A configuration file names a profile and overrides individual fields, for
example ``{"profile": "standard", "scoring": "grouped_oof"}``.  The
``standard`` profile is the budget of the published runs; ``small`` is a
quick profile for tests and trial runs.  ``jobs`` and ``task_size`` only
change how work is split over processes, never the result.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .metrics import SCORING_MODES

ENGINEERING_FIELDS = ("jobs", "task_size")


@dataclass(frozen=True)
class Settings:
    profile: str
    seed: int
    scoring: str
    epsilon: float
    jobs: int
    task_size: int
    initial_component_target: int
    component_population: int
    component_islands: int
    component_min_generations: int
    component_max_generations: int
    component_patience: int
    component_persist_per_generation: int
    component_context_candidates: int
    component_context_anchors: int
    component_context_screen_anchors: int
    component_context_full_audit_candidates: int
    component_context_candidate_block: int
    component_context_anchor_block: int
    component_capacity: int
    component_feedback_reserve: int
    skeleton_population: int
    skeleton_islands: int
    skeleton_min_generations: int
    skeleton_max_generations: int
    skeleton_patience: int
    skeleton_warmup_generations: int
    feedback_interval: int
    feedback_component_proposals: int
    feedback_component_persist: int
    feedback_component_inject: int
    feedback_component_migrate: int
    feedback_side_population: int
    feedback_side_generations: int
    joint_final_generations: int
    beam_topologies: int
    beam_width: int
    performance_archive_cap: int
    topology_archive_cap: int
    prediction_archive_cap: int
    final_candidate_cap: int
    final_top_n: int
    max_attempt_factor: int
    # Stop after a run of generations without improvement, then finish with
    # a short joint consolidation; an improvement during it resumes the search.
    saturation_stop: bool
    saturation_min_generations: int
    saturation_patience: int
    saturation_min_feedback_cycles: int
    saturation_consolidation_generations: int
    # End skeleton evolution after this generation and finalize from the
    # archives reached at that point.
    final_skeleton_generation: int | None

    def scientific(self) -> dict[str, Any]:
        out = asdict(self)
        for name in ENGINEERING_FIELDS:
            out.pop(name)
        return out


_COMMON = {"seed": 56, "scoring": "hard_core", "epsilon": 1.0e-6, "jobs": 1,
           "task_size": 256, "final_top_n": 30, "saturation_stop": False,
           "final_skeleton_generation": None}

PROFILES: dict[str, dict[str, Any]] = {
    "standard": {
        **_COMMON,
        "initial_component_target": 100000, "component_population": 65536,
        "component_islands": 16, "component_min_generations": 48,
        "component_max_generations": 96, "component_patience": 24,
        "component_persist_per_generation": 16000, "component_context_candidates": 12288,
        "component_context_anchors": 192, "component_context_screen_anchors": 24,
        "component_context_full_audit_candidates": 768,
        "component_context_candidate_block": 12, "component_context_anchor_block": 8,
        "component_capacity": 2000000, "component_feedback_reserve": 400000,
        "skeleton_population": 65536, "skeleton_islands": 16,
        "skeleton_min_generations": 96, "skeleton_max_generations": 160,
        "skeleton_patience": 28, "skeleton_warmup_generations": 8,
        "feedback_interval": 4, "feedback_component_proposals": 32768,
        "feedback_component_persist": 12288, "feedback_component_inject": 8192,
        "feedback_component_migrate": 2048, "feedback_side_population": 8192,
        "feedback_side_generations": 3, "joint_final_generations": 24,
        "beam_topologies": 16000, "beam_width": 6, "performance_archive_cap": 75000,
        "topology_archive_cap": 30000, "prediction_archive_cap": 20000,
        "final_candidate_cap": 8192, "max_attempt_factor": 160,
        "saturation_min_generations": 24, "saturation_patience": 12,
        "saturation_min_feedback_cycles": 4, "saturation_consolidation_generations": 6,
    },
    "small": {
        **_COMMON,
        "initial_component_target": 1800, "component_population": 512,
        "component_islands": 4, "component_min_generations": 2,
        "component_max_generations": 2, "component_patience": 2,
        "component_persist_per_generation": 256, "component_context_candidates": 256,
        "component_context_anchors": 16, "component_context_screen_anchors": 8,
        "component_context_full_audit_candidates": 32,
        "component_context_candidate_block": 8, "component_context_anchor_block": 8,
        "component_capacity": 16000, "component_feedback_reserve": 2000,
        "skeleton_population": 512, "skeleton_islands": 4,
        "skeleton_min_generations": 3, "skeleton_max_generations": 3,
        "skeleton_patience": 3, "skeleton_warmup_generations": 1,
        "feedback_interval": 2, "feedback_component_proposals": 256,
        "feedback_component_persist": 128, "feedback_component_inject": 96,
        "feedback_component_migrate": 32, "feedback_side_population": 128,
        "feedback_side_generations": 1, "joint_final_generations": 1,
        "beam_topologies": 128, "beam_width": 3, "performance_archive_cap": 512,
        "topology_archive_cap": 256, "prediction_archive_cap": 256,
        "final_candidate_cap": 256, "max_attempt_factor": 80,
        "saturation_min_generations": 3, "saturation_patience": 2,
        "saturation_min_feedback_cycles": 1, "saturation_consolidation_generations": 1,
    },
}


def make_settings(values: dict[str, Any]) -> Settings:
    values = dict(values)
    values.pop("description", None)
    profile = str(values.pop("profile", "standard"))
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    known = {f.name for f in fields(Settings)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(unknown)}")
    settings = Settings(profile=profile, **{**PROFILES[profile], **values})
    validate(settings)
    return settings


def load_settings(path: Path, **overrides: Any) -> Settings:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    values.update({k: v for k, v in overrides.items() if v is not None})
    return make_settings(values)


def validate(s: Settings) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(s.scoring in SCORING_MODES, f"scoring must be one of {SCORING_MODES}")
    require(s.jobs >= 1 and s.task_size >= 1, "jobs and task_size must be positive")
    require(0 < s.epsilon < 1, "epsilon must lie in (0, 1)")
    require(s.component_capacity >= s.initial_component_target,
            "component capacity below the initial target")
    require(0 <= s.component_feedback_reserve < s.component_capacity,
            "invalid feedback reserve")
    require(s.component_capacity - s.component_feedback_reserve >= s.initial_component_target,
            "ordinary component capacity below the initial target")
    require(s.component_population >= s.component_islands
            and s.skeleton_population >= s.skeleton_islands, "population below island count")
    require(s.component_min_generations <= s.component_max_generations
            and s.skeleton_min_generations <= s.skeleton_max_generations,
            "minimum generations exceed maximum")
    require(s.feedback_interval >= 1 and s.beam_width >= 1, "invalid feedback interval or beam")
    require(1 <= s.component_context_screen_anchors <= s.component_context_anchors,
            "screen anchors outside the anchor panel")
    require(0 <= s.component_context_full_audit_candidates <= s.component_context_candidates,
            "invalid full-audit budget")
    require(s.component_context_candidate_block >= 1 and s.component_context_anchor_block >= 1,
            "invalid context block size")
    require(0 <= s.feedback_component_inject <= s.feedback_component_persist
            <= s.feedback_component_proposals, "feedback budgets out of order")
    require(0 <= s.feedback_component_migrate <= s.feedback_component_inject,
            "feedback migration cap exceeds the injection budget")
    require(s.feedback_side_population >= s.feedback_component_migrate
            and s.feedback_side_generations >= 0, "invalid feedback side budget")
    if s.saturation_stop:
        require(s.skeleton_warmup_generations <= s.saturation_min_generations
                <= s.skeleton_max_generations, "saturation minimum outside the generation range")
        require(s.saturation_patience >= 1 and s.saturation_min_feedback_cycles >= 0
                and s.saturation_consolidation_generations >= 0, "invalid saturation settings")
    if s.final_skeleton_generation is not None:
        require(1 <= s.final_skeleton_generation <= s.skeleton_max_generations,
                "final skeleton generation outside the generation range")
