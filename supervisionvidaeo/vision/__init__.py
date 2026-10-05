"""
Vision Supervision — desde principios fundamentales.
Librería nativa de supervisión por computadora, construida sin dependencias externas.
"""
from .deteccion import Deteccion, LoteDetecciones
from .anotador import Anotador
from .rastreador import RastreadorIOU
from .procesador import ProcesadorVideo
from .metricas import Metricas
from .detectores import DetectorColor, DetectorMovimiento
from .qr_vision import QRDeteccion, LoteQRs, DetectorQR, ProcesadorQRVideo
from .qr_coder import QRCode, QRLote, QRCoder, QRBatchCoder
from .video_supervision import FrameAnalysis, VideoReport, SupervisorVideo, supervisar_video

__version__ = "0.4.0"
__all__ = [
    "Deteccion",
    "LoteDetecciones",
    "Anotador",
    "RastreadorIOU",
    "ProcesadorVideo",
    "Metricas",
    "DetectorColor",
    "DetectorMovimiento",
    "QRDeteccion",
    "LoteQRs",
    "DetectorQR",
    "ProcesadorQRVideo",
    "QRCode",
    "QRLote",
    "QRCoder",
    "QRBatchCoder",
    "FrameAnalysis",
    "VideoReport",
    "SupervisorVideo",
    "supervisar_video",
]
