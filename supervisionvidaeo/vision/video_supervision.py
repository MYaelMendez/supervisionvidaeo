"""
Video Supervision — análisis y verificación de video renderizado.

Integra vision-supervision con el pipeline de producción de video (æRTXrender).
Cada frame renderizado se analiza: luminancia, contraste, movimiento, QR.
Produce un reporte de supervisión (receipt).
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

from .deteccion import Deteccion
from .qr_vision import DetectorQR


@dataclass
class FrameAnalysis:
    """Análisis de un frame individual."""
    index: int
    luminancia: float          # 0-255 mean brightness
    contraste: float           # std dev
    es_negro: bool             # near-black frame
    tiene_qr: bool
    qr_texto: Optional[str] = None
    hash: Optional[str] = None


@dataclass
class VideoReport:
    """Reporte de supervisión de un video."""
    total_frames: int = 0
    frames_negros: int = 0
    frames_con_qr: int = 0
    luminancia_media: float = 0.0
    contraste_medio: float = 0.0
    movimiento_medio: float = 0.0
    calidad: str = "unknown"       # PASS | WARN | FAIL
    razones: List[str] = field(default_factory=list)
    receipt: Optional[str] = None


class SupervisorVideo:
    """Supervisor de video — analiza frames y produce veredicto."""

    # Umbrales
    UMBRAL_NEGRO = 12.0            # luminancia < esto = frame negro
    UMBRAL_LUMINANCIA_MIN = 20.0   # video demasiado oscuro
    UMBRAL_LUMINANCIA_MAX = 235.0  # video quemado
    UMBRAL_CONTRASTE_MIN = 8.0     # video plano/sin detalle
    MAX_FRAMES_NEGROS_PCT = 0.05   # >5% negros = FAIL
    MAX_FRAMES_NEGROS_WARN = 0.01  # >1% negros = WARN

    def __init__(self, detectar_qr: bool = False):
        self.detectar_qr = detectar_qr
        self.detector_qr = DetectorQR(usar_tracking=False) if detectar_qr else None

    def analizar_frame(self, frame: np.ndarray, index: int = 0) -> FrameAnalysis:
        """Analizar un frame individual."""
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lum = float(gris.mean())
        contraste = float(gris.std())
        es_negro = lum < self.UMBRAL_NEGRO

        tiene_qr = False
        qr_texto = None
        if self.detectar_qr and self.detector_qr and not es_negro:
            lote = self.detector_qr.detectar(frame)
            if lote.qrs:
                tiene_qr = True
                qr_texto = lote.qrs[0].texto[:64]

        frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()[:16]

        return FrameAnalysis(
            index=index,
            luminancia=lum,
            contraste=contraste,
            es_negro=es_negro,
            tiene_qr=tiene_qr,
            qr_texto=qr_texto,
            hash=frame_hash,
        )

    def supervisar_frames(self, frames_dir: str, muestrear: int = 30) -> VideoReport:
        """Supervisar un directorio de frames PNG.
        
        Args:
            frames_dir: directorio con frame_*.png
            muestrear: analizar 1 de cada N frames (para velocidad)
            
        Returns:
            VideoReport con veredicto
        """
        frames_dir = Path(frames_dir)
        archivos = sorted(frames_dir.glob("frame_*.png"))
        if not archivos:
            report = VideoReport(calidad="FAIL")
            report.razones.append("no frames found")
            return report

        report = VideoReport()
        lums, contrastes = [], []
        prev_gris = None
        movimientos = []
        analizados = 0

        for i, path in enumerate(archivos):
            if i % muestrear != 0 and i != len(archivos) - 1:
                continue

            frame = cv2.imread(str(path))
            if frame is None:
                continue

            fa = self.analizar_frame(frame, i)
            analizados += 1
            lums.append(fa.luminancia)
            contrastes.append(fa.contraste)
            if fa.es_negro:
                report.frames_negros += 1
            if fa.tiene_qr:
                report.frames_con_qr += 1

            # Movimiento entre frames analizados
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gris_small = cv2.resize(gris, (160, 284))
            if prev_gris is not None:
                diff = cv2.absdiff(gris_small, prev_gris)
                movimientos.append(float(diff.mean()))
            prev_gris = gris_small

        report.total_frames = analizados
        report.luminancia_media = float(np.mean(lums)) if lums else 0.0
        report.contraste_medio = float(np.mean(contrastes)) if contrastes else 0.0
        report.movimiento_medio = float(np.mean(movimientos)) if movimientos else 0.0

        # Veredicto
        pct_negros = report.frames_negros / analizados if analizados else 1.0

        if pct_negros > self.MAX_FRAMES_NEGROS_PCT:
            report.calidad = "FAIL"
            report.razones.append(f"{report.frames_negros}/{analizados} frames negros ({pct_negros*100:.1f}%)")
        elif report.luminancia_media < self.UMBRAL_LUMINANCIA_MIN:
            report.calidad = "FAIL"
            report.razones.append(f"luminancia media muy baja ({report.luminancia_media:.1f})")
        elif report.contraste_medio < self.UMBRAL_CONTRASTE_MIN:
            report.calidad = "FAIL"
            report.razones.append(f"contraste muy bajo ({report.contraste_medio:.1f}) — video plano")
        elif pct_negros > self.MAX_FRAMES_NEGROS_WARN:
            report.calidad = "WARN"
            report.razones.append(f"{report.frames_negros}/{analizados} frames negros ({pct_negros*100:.1f}%)")
        elif report.movimiento_medio < 0.5:
            report.calidad = "WARN"
            report.razones.append(f"movimiento muy bajo ({report.movimiento_medio:.2f}) — video estático")
        else:
            report.calidad = "PASS"

        # Receipt
        receipt_data = {
            "total_frames": report.total_frames,
            "frames_negros": report.frames_negros,
            "luminancia_media": round(report.luminancia_media, 2),
            "contraste_medio": round(report.contraste_medio, 2),
            "movimiento_medio": round(report.movimiento_medio, 3),
            "calidad": report.calidad,
        }
        report.receipt = hashlib.sha256(
            json.dumps(receipt_data, sort_keys=True).encode()
        ).hexdigest()

        return report


def supervisar_video(video_path: str, detectar_qr: bool = False, muestrear: int = 30) -> dict:
    """Supervisar un archivo MP4 extrayendo frames con OpenCV.
    
    Args:
        video_path: ruta al MP4
        detectar_qr: si analizar QRs en los frames
        muestrear: analizar 1 de cada N frames
        
    Returns:
        dict con el reporte
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"cannot open {video_path}", "calidad": "FAIL"}

    supervisor = SupervisorVideo(detectar_qr=detectar_qr)
    lums, contrastes = [], []
    prev_gris = None
    movimientos = []
    frames_negros = 0
    frames_con_qr = 0
    analizados = 0
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % muestrear == 0:
            fa = supervisor.analizar_frame(frame, i)
            analizados += 1
            lums.append(fa.luminancia)
            contrastes.append(fa.contraste)
            if fa.es_negro:
                frames_negros += 1
            if fa.tiene_qr:
                frames_con_qr += 1
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gris_small = cv2.resize(gris, (160, 284))
            if prev_gris is not None:
                movimientos.append(float(cv2.absdiff(gris_small, prev_gris).mean()))
            prev_gris = gris_small
        i += 1

    cap.release()

    lum_media = float(np.mean(lums)) if lums else 0.0
    contraste_medio = float(np.mean(contrastes)) if contrastes else 0.0
    mov_medio = float(np.mean(movimientos)) if movimientos else 0.0
    pct_negros = frames_negros / analizados if analizados else 1.0

    razones = []
    if pct_negros > 0.05:
        calidad = "FAIL"
        razones.append(f"{frames_negros}/{analizados} frames negros ({pct_negros*100:.1f}%)")
    elif lum_media < 20:
        calidad = "FAIL"
        razones.append(f"luminancia baja ({lum_media:.1f})")
    elif contraste_medio < 8:
        calidad = "FAIL"
        razones.append(f"contraste bajo ({contraste_medio:.1f})")
    elif pct_negros > 0.01 or mov_medio < 0.5:
        calidad = "WARN"
        if pct_negros > 0.01:
            razones.append(f"{frames_negros} frames negros")
        if mov_medio < 0.5:
            razones.append(f"movimiento bajo ({mov_medio:.2f})")
    else:
        calidad = "PASS"

    receipt = hashlib.sha256(json.dumps({
        "analizados": analizados, "negros": frames_negros,
        "lum": round(lum_media, 2), "contraste": round(contraste_medio, 2),
        "movimiento": round(mov_medio, 3), "calidad": calidad,
    }, sort_keys=True).encode()).hexdigest()

    return {
        "video": str(video_path),
        "frames_analizados": analizados,
        "frames_negros": frames_negros,
        "frames_con_qr": frames_con_qr,
        "luminancia_media": round(lum_media, 2),
        "contraste_medio": round(contraste_medio, 2),
        "movimiento_medio": round(mov_medio, 3),
        "calidad": calidad,
        "razones": razones,
        "receipt": receipt,
        "supervisado_at": datetime.now(timezone.utc).isoformat(),
    }
