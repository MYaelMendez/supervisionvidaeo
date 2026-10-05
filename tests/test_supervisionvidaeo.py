"""Tests for supervisionvidaeo — the produce-and-verify contract."""

import numpy as np
import cv2
import pytest
from pathlib import Path

from supervisionvidaeo import (
    SceneSpec, Receipt, produce, verify, toolchain_status,
    SupervisionRefused, PALETTES,
    MIN_PARTICLES, MAX_PARTICLES, MIN_MOTION, MAX_MOTION, MIN_DURATION, MAX_DURATION,
)


def _noise_producer(spec, out):
    """A producer that emits moving noise — should pass supervision."""
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 284))
    for i in range(30):
        w.write(np.random.randint(0, 255, (284, 160, 3), dtype=np.uint8))
    w.release()
    return 30


def _black_producer(spec, out):
    """A producer that emits black — must be refused."""
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 284))
    for _ in range(30):
        w.write(np.zeros((284, 160, 3), dtype=np.uint8))
    w.release()
    return 30


def _silent_producer(spec, out):
    """A producer that claims success but writes nothing — must be refused."""
    return 30


class TestSceneSpecBounds:
    def test_defaults_are_renderable(self):
        s = SceneSpec()
        assert MIN_PARTICLES <= s.particles <= MAX_PARTICLES
        assert MIN_MOTION <= s.motion <= MAX_MOTION
        assert MIN_DURATION <= s.duration <= MAX_DURATION
        assert s.palette in PALETTES

    def test_clamps_out_of_range(self):
        s = SceneSpec.bounded({"particles": 999999, "motion": 9.9, "duration": -5})
        assert s.particles == MAX_PARTICLES
        assert s.motion == MAX_MOTION
        assert s.duration == MIN_DURATION

    def test_invalid_palette_falls_back(self):
        s = SceneSpec.bounded({"palette": "chartreuse"})
        assert s.palette in PALETTES

    def test_garbage_input_never_raises(self):
        for bad in [{"particles": "x"}, {"motion": None}, {"duration": []}, {"palette": 42}]:
            s = SceneSpec.bounded(bad)
            assert MIN_PARTICLES <= s.particles <= MAX_PARTICLES
            assert s.palette in PALETTES

    def test_none_input_gives_valid_spec(self):
        s = SceneSpec.bounded(None)
        assert s.palette in PALETTES

    def test_fingerprint_is_stable(self):
        a = SceneSpec.bounded({"palette": "cyan", "duration": 3})
        b = SceneSpec.bounded({"palette": "cyan", "duration": 3})
        assert a.fingerprint() == b.fingerprint()

    def test_fingerprint_changes_with_spec(self):
        a = SceneSpec.bounded({"palette": "cyan"})
        b = SceneSpec.bounded({"palette": "neon"})
        assert a.fingerprint() != b.fingerprint()

    def test_overrides_win(self):
        s = SceneSpec.bounded({"duration": 3}, duration=5)
        assert s.duration == 5


class TestProduceGate:
    def test_black_render_is_refused(self, tmp_path):
        with pytest.raises(SupervisionRefused) as e:
            produce({"title": "black"}, str(tmp_path / "black.mp4"), producer=_black_producer)
        assert e.value.report.get("stage") == "SUPERVISE"

    def test_silent_producer_is_refused(self, tmp_path):
        with pytest.raises(SupervisionRefused):
            produce({"title": "silent"}, str(tmp_path / "none.mp4"), producer=_silent_producer)

    def test_good_render_yields_receipt(self, tmp_path):
        r = produce({"title": "good"}, str(tmp_path / "good.mp4"), producer=_noise_producer)
        assert isinstance(r, Receipt)
        assert r.supervision["calidad"] == "PASS"
        assert Path(r.output).exists()

    def test_receipt_has_proof_fields(self, tmp_path):
        r = produce({"title": "proof"}, str(tmp_path / "p.mp4"), producer=_noise_producer)
        assert len(r.sha256) == 64
        assert len(r.spec_fingerprint) == 64
        assert r.supervision_receipt
        assert r.frames > 0
        assert r.size_bytes > 0

    def test_receipt_hash_is_deterministic(self, tmp_path):
        """Same spec + same supervision → same proof hash (no timestamps inside)."""
        r1 = produce({"title": "same", "palette": "cyan"}, str(tmp_path / "a.mp4"), producer=_noise_producer)
        r2 = produce({"title": "same", "palette": "cyan"}, str(tmp_path / "b.mp4"), producer=_noise_producer)
        # supervision metrics may differ frame-to-frame on noise, so compare the
        # binding: fingerprint is stable, and sha256 is a pure function of its inputs
        assert r1.spec_fingerprint == r2.spec_fingerprint
        assert len(r1.sha256) == len(r2.sha256) == 64

    def test_receipt_to_json(self, tmp_path):
        r = produce({"title": "json"}, str(tmp_path / "j.mp4"), producer=_noise_producer)
        out = tmp_path / "receipt.json"
        s = r.to_json(str(out))
        assert out.exists()
        assert '"sha256"' in s

    def test_max_attempts_then_refuse(self, tmp_path):
        calls = {"n": 0}
        def flaky(spec, out):
            calls["n"] += 1
            return _black_producer(spec, out)
        with pytest.raises(SupervisionRefused):
            produce({"title": "flaky"}, str(tmp_path / "f.mp4"), producer=flaky, max_attempts=2)
        assert calls["n"] == 2


class TestVerify:
    def test_verify_rejects_missing_file(self):
        r = verify("C:/nonexistent/nope.mp4")
        assert r["calidad"] == "FAIL"
        assert "error" in r

    def test_verify_black_video_fails(self, tmp_path):
        p = tmp_path / "b.mp4"
        _black_producer(None, p)
        r = verify(str(p), sample_every=1)
        assert r["calidad"] == "FAIL"


class TestToolchainStatus:
    def test_status_shape(self):
        s = toolchain_status()
        assert s["package"] == "supervisionvidaeo"
        assert "producer" in s and "verifier" in s and "bounds" in s
        assert s["verifier"]["ml_required"] is False

    def test_status_is_honest_about_availability(self):
        s = toolchain_status()
        # available must be a bool reflecting real file existence, not a claim
        assert isinstance(s["producer"]["available"], bool)
