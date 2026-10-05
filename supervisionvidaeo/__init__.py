"""
supervisionvidaeo — produce and verify in one contract.

Most generative pipelines hand you an output and hope. This one refuses to hand
you an output it has not measured. The producer (vidæo) and the verifier
(supervision) are not two libraries you wire together — they are one loop:

    SPEC → PRODUCE → SUPERVISE → (PASS: emit receipt) | (FAIL: refuse)

The public entry point is :func:`produce`. It returns a :class:`Receipt` only
when the render passes supervision. A failed render raises
:class:`SupervisionRefused` — never a plausible-looking path to a broken file.

Design rules (learned in practice):
  - A stage advances when its gate passes, never on a timer.
  - The producer proposes; bounds are enforced before a frame renders.
  - The verifier is deterministic (statistics + geometry), not ML — every
    verdict traces to a measurable threshold, so it is auditable.
  - A receipt is proof, not decoration: same render → same receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .vision import supervisar_video

__version__ = "1.0.0"

# ── the bounded spec space ──
PALETTES = ("gold", "cyan", "neon", "amber", "mono")
MIN_PARTICLES, MAX_PARTICLES = 2000, 20000
MIN_MOTION, MAX_MOTION = 0.01, 0.08
MIN_DURATION, MAX_DURATION = 2, 6
DEFAULT_FPS = 30


class SupervisionRefused(Exception):
    """Raised when a render fails its gate. The output is NOT returned."""

    def __init__(self, reason: str, report: Optional[Dict[str, Any]] = None):
        super().__init__(reason)
        self.reason = reason
        self.report = report or {}


@dataclass
class SceneSpec:
    """A bounded, renderable scene description. Construct via :meth:`bounded`."""
    title: str = "untitled"
    particles: int = 6000
    palette: str = "gold"
    motion: float = 0.03
    duration: int = 4
    fps: int = DEFAULT_FPS
    width: int = 720
    height: int = 1280

    @classmethod
    def bounded(cls, raw: Optional[Dict[str, Any]] = None, **overrides: Any) -> "SceneSpec":
        """Coerce anything into a renderable spec. The model proposes intent;
        this guarantees the values are renderable. Never raises on bad input."""
        raw = {**(raw or {}), **overrides}
        base = cls()

        def _int(v, default, lo, hi):
            try:
                return max(lo, min(hi, int(v)))
            except (TypeError, ValueError):
                return default

        def _float(v, default, lo, hi):
            try:
                return max(lo, min(hi, float(v)))
            except (TypeError, ValueError):
                return default

        pal = str(raw.get("palette", base.palette)).lower().strip()
        if pal not in PALETTES:
            pal = base.palette

        return cls(
            title=str(raw.get("title", base.title))[:48],
            particles=_int(raw.get("particles"), base.particles, MIN_PARTICLES, MAX_PARTICLES),
            palette=pal,
            motion=_float(raw.get("motion"), base.motion, MIN_MOTION, MAX_MOTION),
            duration=_int(raw.get("duration"), base.duration, MIN_DURATION, MAX_DURATION),
            fps=_int(raw.get("fps"), DEFAULT_FPS, 1, 120),
            width=_int(raw.get("width"), base.width, 16, 4096),
            height=_int(raw.get("height"), base.height, 16, 4096),
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable hash of the spec — the producer's input identity."""
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode()
        ).hexdigest()


@dataclass
class Receipt:
    """Proof that a render was produced AND passed supervision.

    ``sha256`` binds the spec fingerprint to the supervision receipt, so the
    same scene always yields the same proof and any change to either side
    changes it. This is the artifact a downstream consumer trusts.
    """
    spec: Dict[str, Any]
    spec_fingerprint: str
    output: str
    frames: int
    size_bytes: int
    supervision: Dict[str, Any]      # calidad, luminancia, contraste, movimiento, negros
    supervision_receipt: str
    sha256: str = ""
    produced_at: str = ""

    def __post_init__(self):
        if not self.sha256:
            payload = json.dumps({
                "spec_fingerprint": self.spec_fingerprint,
                "supervision_receipt": self.supervision_receipt,
                "frames": self.frames,
                "size_bytes": self.size_bytes,
            }, sort_keys=True)
            self.sha256 = hashlib.sha256(payload.encode()).hexdigest()
        if not self.produced_at:
            self.produced_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(self.as_dict(), indent=2, ensure_ascii=False)
        if path:
            Path(path).write_text(s, encoding="utf-8")
        return s


# ── the producer, injected ──
# A producer takes a SceneSpec and an output path, renders frames, and returns
# the frame count. It is a plain callable so the loop is testable without a GPU.
ProducerFn = Callable[[SceneSpec, Path], int]


def _default_producer(spec: SceneSpec, out: Path) -> int:
    """Render via the local vidæo toolchain (CuPy on the operator's GPU)."""
    vidaeo_py = Path(r"C:\æ\vidæo\vidæo.py")
    gpu_py = r"C:\gpu\Scripts\python.exe"
    if not (vidaeo_py.exists() and Path(gpu_py).exists()):
        raise SupervisionRefused(
            "no producer available: C:\\æ\\vidæo\\vidæo.py or the GPU python is missing",
            {"stage": "PRODUCE"},
        )
    spec_file = out.with_suffix(".spec.json")
    spec_file.write_text(json.dumps(spec.as_dict()), encoding="utf-8")
    r = subprocess.run(
        [gpu_py, str(vidaeo_py), "cycle", str(spec_file), str(out)],
        capture_output=True, text=True, timeout=900, cwd=str(vidaeo_py.parent),
    )
    if r.returncode != 0 or not out.exists():
        raise SupervisionRefused(
            f"producer failed: {(r.stderr or r.stdout)[-300:]}", {"stage": "PRODUCE"}
        )
    return int(spec.duration * spec.fps)


def produce(
    raw_spec: Optional[Dict[str, Any]] = None,
    out_path: Optional[str] = None,
    *,
    producer: Optional[ProducerFn] = None,
    sample_every: int = 20,
    max_attempts: int = 1,
) -> Receipt:
    """Produce a video and refuse to return it unless it passes supervision.

    Args:
        raw_spec: anything; coerced into a bounded SceneSpec.
        out_path: where the MP4 goes (default: cache/generated/videos/).
        producer: render callable; defaults to the local vidæo toolchain.
        sample_every: supervise 1 of every N frames.
        max_attempts: re-produce on FAIL up to this many times.

    Returns:
        A Receipt (only on PASS).

    Raises:
        SupervisionRefused: if every attempt fails its gate.
    """
    spec = SceneSpec.bounded(raw_spec)
    producer = producer or _default_producer

    if out_path is None:
        cache = Path(os.environ.get(
            "HERMES_HOME",
            str(Path.home() / "AppData" / "Local" / "hermes"),
        )) / "cache" / "generated" / "videos"
        cache.mkdir(parents=True, exist_ok=True)
        out_path = str(cache / f"supervisionvidaeo_{int(time.time())}.mp4")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    last_reason, last_report = "no attempt made", {}
    for attempt in range(1, max_attempts + 1):
        # PRODUCE
        frames = producer(spec, out)
        if not out.exists():
            last_reason = "producer returned but produced no file"
            last_report = {"stage": "PRODUCE", "attempt": attempt}
            continue

        # SUPERVISE — the gate
        report = supervisar_video(str(out), muestrear=sample_every)
        if report.get("calidad") != "PASS":
            last_reason = (
                f"{report.get('calidad')}: {'; '.join(report.get('razones', ['no reason given']))}"
            )
            last_report = {"stage": "SUPERVISE", "attempt": attempt, **report}
            continue

        # PASS — emit the proof
        return Receipt(
            spec=spec.as_dict(),
            spec_fingerprint=spec.fingerprint(),
            output=str(out),
            frames=frames,
            size_bytes=out.stat().st_size,
            supervision={
                "calidad": report["calidad"],
                "luminancia": report["luminancia_media"],
                "contraste": report["contraste_medio"],
                "movimiento": report["movimiento_medio"],
                "frames_negros": report["frames_negros"],
            },
            supervision_receipt=report["receipt"],
        )

    raise SupervisionRefused(last_reason, last_report)


def verify(mp4_path: str, sample_every: int = 20) -> Dict[str, Any]:
    """Supervise an existing video. The gate, without the producer."""
    return supervisar_video(mp4_path, muestrear=sample_every)


def toolchain_status() -> Dict[str, Any]:
    """Report whether this package can actually produce — never assume."""
    vidaeo_py = Path(r"C:\æ\vidæo\vidæo.py")
    gpu_py = Path(r"C:\gpu\Scripts\python.exe")
    return {
        "package": "supervisionvidaeo",
        "version": __version__,
        "producer": {
            "vidaeo_cli": str(vidaeo_py) if vidaeo_py.exists() else None,
            "gpu_python": str(gpu_py) if gpu_py.exists() else None,
            "available": vidaeo_py.exists() and gpu_py.exists(),
        },
        "verifier": {"opencv": True, "ml_required": False},
        "bounds": {
            "particles": [MIN_PARTICLES, MAX_PARTICLES],
            "motion": [MIN_MOTION, MAX_MOTION],
            "duration": [MIN_DURATION, MAX_DURATION],
            "palettes": list(PALETTES),
        },
    }
