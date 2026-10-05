"""Tests para Video Supervision — análisis y verificación de video."""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path

from supervisionvidaeo.vision.video_supervision import (
    FrameAnalysis,
    VideoReport,
    SupervisorVideo,
    supervisar_video,
)


def _frame(color=128, size=(284, 160)):
    """Helper: crear un frame BGR sólido."""
    f = np.zeros((*size, 3), dtype=np.uint8)
    f[:] = color
    return f


def _frame_noise(size=(284, 160)):
    """Helper: frame con ruido (contraste real)."""
    return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)


class TestFrameAnalysis:
    def test_creacion(self):
        fa = FrameAnalysis(index=0, luminancia=128.0, contraste=50.0, es_negro=False, tiene_qr=False)
        assert fa.index == 0
        assert fa.luminancia == 128.0
        assert fa.es_negro is False


class TestVideoReport:
    def test_defaults(self):
        r = VideoReport()
        assert r.total_frames == 0
        assert r.calidad == "unknown"
        assert r.razones == []


class TestSupervisorVideo:
    def test_frame_negro_detectado(self):
        sup = SupervisorVideo()
        fa = sup.analizar_frame(_frame(0), 0)
        assert fa.es_negro is True
        assert fa.luminancia < 12

    def test_frame_brillante_no_negro(self):
        sup = SupervisorVideo()
        fa = sup.analizar_frame(_frame(200), 0)
        assert fa.es_negro is False
        assert fa.luminancia > 100

    def test_contraste_de_ruido(self):
        sup = SupervisorVideo()
        fa = sup.analizar_frame(_frame_noise(), 0)
        # ruido uniforme 0-255 por canal → al pasar a gris, std ~49
        assert fa.contraste > 40

    def test_contraste_de_solido_es_cero(self):
        sup = SupervisorVideo()
        fa = sup.analizar_frame(_frame(128), 0)
        assert fa.contraste < 1.0

    def test_hash_es_deterministico(self):
        sup = SupervisorVideo()
        f = _frame(128)
        a = sup.analizar_frame(f, 0)
        b = sup.analizar_frame(f, 0)
        assert a.hash == b.hash

    def test_hash_difiere_entre_frames(self):
        sup = SupervisorVideo()
        a = sup.analizar_frame(_frame(128), 0)
        b = sup.analizar_frame(_frame(129), 1)
        assert a.hash != b.hash

    def test_supervisar_frames_dir_vacio(self):
        sup = SupervisorVideo()
        with tempfile.TemporaryDirectory() as tmp:
            r = sup.supervisar_frames(tmp)
            assert r.calidad == "FAIL"
            assert "no frames found" in r.razones

    def test_supervisar_frames_negros_es_fail(self):
        sup = SupervisorVideo()
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(10):
                cv2.imwrite(f"{tmp}/frame_{i:05d}.png", _frame(0))
            r = sup.supervisar_frames(tmp, muestrear=1)
            assert r.calidad == "FAIL"
            assert r.frames_negros == 10

    def test_supervisar_frames_ruido_es_pass(self):
        sup = SupervisorVideo()
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(10):
                cv2.imwrite(f"{tmp}/frame_{i:05d}.png", _frame_noise())
            r = sup.supervisar_frames(tmp, muestrear=1)
            # ruido tiene contraste y cambia entre frames → PASS
            assert r.calidad == "PASS"
            assert r.frames_negros == 0

    def test_receipt_es_sha256(self):
        sup = SupervisorVideo()
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                cv2.imwrite(f"{tmp}/frame_{i:05d}.png", _frame_noise())
            r = sup.supervisar_frames(tmp, muestrear=1)
            assert r.receipt is not None
            assert len(r.receipt) == 64  # sha256 hex

    def test_solido_estatico_es_warn(self):
        sup = SupervisorVideo()
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(10):
                cv2.imwrite(f"{tmp}/frame_{i:05d}.png", _frame(200))
            r = sup.supervisar_frames(tmp, muestrear=1)
            # sólido: contraste 0 → FAIL por contraste bajo
            assert r.calidad == "FAIL"
            assert any("contraste" in x for x in r.razones)


class TestSupervisarVideo:
    def test_archivo_inexistente(self):
        r = supervisar_video("C:/nonexistent/fake.mp4")
        assert r["calidad"] == "FAIL"
        assert "error" in r

    def test_video_real_generado(self):
        """Generar un video de prueba y supervisarlo."""
        with tempfile.TemporaryDirectory() as tmp:
            vid_path = str(Path(tmp) / "test.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(vid_path, fourcc, 30, (160, 284))
            for i in range(30):
                frame = _frame_noise()
                cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                out.write(frame)
            out.release()

            r = supervisar_video(vid_path, muestrear=5)
            assert r["calidad"] in ("PASS", "WARN")
            assert r["frames_analizados"] > 0
            assert len(r["receipt"]) == 64

    def test_video_negro_es_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            vid_path = str(Path(tmp) / "black.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(vid_path, fourcc, 30, (160, 284))
            for i in range(30):
                out.write(_frame(0))
            out.release()

            r = supervisar_video(vid_path, muestrear=5)
            assert r["calidad"] == "FAIL"
            assert r["frames_negros"] > 0
