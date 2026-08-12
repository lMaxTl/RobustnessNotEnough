"""M9 - result schema and append-only JSONL writers (spec.md S12).

Analysis reads only these files. A missing field silently produces a wrong
figure, so every field is required and validated on write; `tau` is the only
nullable one (victim `Standard` has AA = 0, spec.md S9.3).

Writes are append-only and flushed + fsynced per record, so a crashed sweep
loses at most one run.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = [
    "ResultRecord", "MarginRecord", "DefenseRecord",
    "JsonlWriter", "read_jsonl", "git_sha", "gpu_name", "utc_now", "new_run_id",
    "RESULTS_PATH", "MARGINS_PATH", "DEFENSES_PATH",
]

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_PATH = RESULTS_DIR / "results.jsonl"
MARGINS_PATH = RESULTS_DIR / "margins.jsonl"
DEFENSES_PATH = RESULTS_DIR / "defenses.jsonl"


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------

def git_sha(short: bool = True) -> str:
    """Current commit, or 'nogit' outside a repository / 'dirty-<sha>' if unclean."""
    root = Path(__file__).resolve().parents[1]
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        sha = subprocess.check_output(
            cmd, cwd=root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        return f"dirty-{sha}" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "nogit"


def gpu_name() -> str:
    """Human-readable accelerator name for the record, e.g. 'NVIDIA GeForce RTX 4060'."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch optional for analysis-only use
        pass
    return f"cpu-{platform.machine()}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return str(uuid.uuid4())


# --------------------------------------------------------------------------
# S12 record
# --------------------------------------------------------------------------

@dataclass
class ResultRecord:
    """One extraction run. Field order matches spec.md S12."""

    victim_name: str
    victim_params: int
    victim_clean_acc: float
    victim_aa_acc: float
    is_control: bool
    student_arch: str
    student_params: int
    attack: str
    pool: str
    defense: str
    defense_params: dict[str, Any]
    n_queries_nominal: int
    n_queries_actual: int
    epochs: int
    seed: int
    student_clean_acc: float
    student_aa_acc: float
    student_pgd40_acc: float
    fidelity_clean: float
    fidelity_adv: float
    tau: float | None
    n_eval: int
    c_query: int
    c_synthesis: int
    c_student: int
    c_total: int
    c_query_heuristic: int
    c_student_heuristic: int
    wallclock_sec: float
    gpu: str

    run_id: str = field(default_factory=new_run_id)
    timestamp: str = field(default_factory=utc_now)
    git_sha: str = field(default_factory=git_sha)
    # Auxiliary, never read by the primary analysis (e.g. prada_detect_index,
    # temperature of a control victim, torch version).
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "ResultRecord":
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None and f.name != "tau":
                raise ValueError(f"field {f.name!r} is required and must not be None")
        for name in ("victim_clean_acc", "victim_aa_acc", "student_clean_acc",
                     "student_aa_acc", "student_pgd40_acc", "fidelity_clean",
                     "fidelity_adv"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name}={v} is not a fraction in [0, 1]")
        if self.n_eval not in (1000, 10000):
            raise ValueError(f"n_eval must be 1000 (sweep) or 10000 (final), got {self.n_eval}")
        # Pitfall 4: AutoAttack is never weaker than PGD-40. If it looks weaker,
        # the student is gradient-masking or the evaluation is broken.
        if self.student_aa_acc > self.student_pgd40_acc + 1e-9:
            raise ValueError(
                f"student_aa_acc ({self.student_aa_acc}) > student_pgd40_acc "
                f"({self.student_pgd40_acc}): AutoAttack must be at least as strong as PGD-40"
            )
        if self.c_total != self.c_query + self.c_synthesis + self.c_student:
            raise ValueError("c_total must equal c_query + c_synthesis + c_student")
        if self.tau is not None and self.victim_aa_acc <= 0:
            raise ValueError("tau must be null when the victim has AA accuracy 0 (spec.md S9.3)")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=False)


@dataclass
class MarginRecord:
    """One victim's boundary geometry (spec.md S10.3) -> margins.jsonl."""

    victim_name: str
    victim_aa_acc: float
    n_images: int
    eps_ref: float                  # 8/255
    margins: list[float]            # per-image linf margin, len == n_images
    censored: list[bool]            # True where FAB failed to flip within budget
    median_margin: float | None     # Kaplan-Meier median; null if too censored
    iqr: list[float | None]         # [q25, q75], Kaplan-Meier
    frac_margin_le_8_255: float
    rho_8_255: float | None         # near-boundary density estimate
    fab_n_restarts: int
    fab_n_iter: int
    search_bound: float
    wallclock_sec: float
    gpu: str
    run_id: str = field(default_factory=new_run_id)
    timestamp: str = field(default_factory=utc_now)
    git_sha: str = field(default_factory=git_sha)

    def validate(self) -> "MarginRecord":
        if len(self.margins) != self.n_images or len(self.censored) != self.n_images:
            raise ValueError("margins/censored length must equal n_images")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=False)


@dataclass
class DefenseRecord:
    """Per-defense attacker-cost multiplier and defender utility cost (spec.md S11.4)."""

    victim_name: str
    defense: str
    defense_params: dict[str, Any]
    victim_clean_acc_none: float
    victim_clean_acc_defended: float
    delta_clean_acc: float          # utility cost, in accuracy points (fraction)
    student_aa_acc_none: float      # baseline student AA under `none`
    student_aa_acc_defended: float
    c_att_none: float
    c_att_defended: float           # cost to reach student_aa_acc_none under d
    c_att_censored: bool            # True when unreachable within 10x max budget
    log10_cost_multiplier: float
    cost_effectiveness: float | None  # orders of magnitude per point of utility
    n_seeds: int
    detect_index: int | None        # prada: query index at which detection fired
    wallclock_sec: float
    gpu: str
    run_id: str = field(default_factory=new_run_id)
    timestamp: str = field(default_factory=utc_now)
    git_sha: str = field(default_factory=git_sha)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=False)


# --------------------------------------------------------------------------
# Append-only IO
# --------------------------------------------------------------------------

class JsonlWriter:
    """Append-only JSONL sink; flush + fsync per record (spec.md S12)."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record) -> None:
        if hasattr(record, "validate"):
            record.validate()
        line = record.to_json() if hasattr(record, "to_json") else json.dumps(record)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def extend(self, records: Iterable) -> None:
        for r in records:
            self.append(r)


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # truncated final line after a crash
                raise ValueError(f"{p}:{lineno}: malformed JSON ({exc})") from exc
