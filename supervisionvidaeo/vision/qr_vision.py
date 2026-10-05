"""
QR Vision — detección y decodificación de códigos QR en tiempo real.

Integra OpenCV QRCodeDetector con el sistema de supervisión por computadora.
Conecta con contract_qr.py para decodificación de contratos inteligentes.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json
import hashlib

from .deteccion import Deteccion
from .anotador import Anotador
from .rastreador import RastreadorIOU


@dataclass
class QRDeteccion:
    """Detección de un código QR individual."""
    texto: str
    puntos: np.ndarray  # 4 esquinas del QR
    centro: tuple
    confianza: float = 1.0
    contrato: Optional[dict] = None
    hash: Optional[str] = None


@dataclass
class LoteQRs:
    """Lote de QRs detectados en un frame."""
    qrs: list = field(default_factory=list)
    frame_id: int = 0


class DetectorQR:
    """Detector de códigos QR usando OpenCV QRCodeDetector.
    
    Integra con el sistema de supervisión para tracking y anotación.
    """
    
    def __init__(self, usar_tracking: bool = True):
        self.detector = cv2.QRCodeDetector()
        self.rastreador = RastreadorIOU() if usar_tracking else None
        self.anotador = Anotador()
        self._frame_count = 0
        self._qr_history: dict = {}  # id -> historial de detecciones
    
    def detectar(self, frame: np.ndarray) -> LoteQRs:
        """Detectar QRs en un frame.
        
        Args:
            frame: Imagen BGR (numpy array)
            
        Returns:
            LoteQRs con todos los QRs detectados
        """
        self._frame_count += 1
        lote = LoteQRs(frame_id=self._frame_count)
        
        # Detectar múltiples QRs
        ret, decoded_info, points, _ = self.detector.detectAndDecodeMulti(frame)
        
        if not ret or points is None:
            return lote
        
        for i, (texto, pts) in enumerate(zip(decoded_info, points)):
            if not texto:
                continue
            
            pts = pts.astype(np.float32)
            centro = tuple(pts.mean(axis=0).astype(int))
            
            # Calcular confianza basada en tamaño y forma
            area = cv2.contourArea(pts.astype(np.int32))
            confianza = min(1.0, area / 5000.0)
            
            # Intentar parsear como contrato
            contrato = None
            hash_qr = None
            try:
                parsed = json.loads(texto)
                if isinstance(parsed, dict) and "type" in parsed:
                    contrato = parsed
                    hash_qr = parsed.get("hash")
                elif isinstance(parsed, dict) and "c" in parsed:
                    # QR chunked
                    hash_qr = parsed.get("h")
            except (json.JSONDecodeError, TypeError):
                pass
            
            deteccion = QRDeteccion(
                texto=texto,
                puntos=pts,
                centro=centro,
                confianza=confianza,
                contrato=contrato,
                hash=hash_qr
            )
            
            # Tracking
            if self.rastreador:
                track_id = self.rastreador.actualizar(
                    Deteccion(
                        etiqueta=f"QR_{hash_qr[:8] if hash_qr else 'unknown'}",
                        confianza=confianza,
                        bbox=self._pts_to_bbox(pts)
                    )
                )
                deteccion.track_id = track_id
            
            lote.qrs.append(deteccion)
        
        return lote
    
    def _pts_to_bbox(self, pts: np.ndarray) -> tuple:
        """Convertir 4 puntos a bbox (x, y, w, h)."""
        x_min = int(pts[:, 0].min())
        y_min = int(pts[:, 1].min())
        x_max = int(pts[:, 0].max())
        y_max = int(pts[:, 1].max())
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def anotar(self, frame: np.ndarray, lote: LoteQRs) -> np.ndarray:
        """Anotar frame con QRs detectados.
        
        Args:
            frame: Imagen BGR original (no modificada)
            lote: LoteQRs de la detección
            
        Returns:
            Frame anotado (copia)
        """
        frame_anotado = frame.copy()
        for qr in lote.qrs:
            # Dibujar contorno
            pts = qr.puntos.astype(np.int32)
            cv2.polylines(frame_anotado, [pts], True, (0, 255, 0), 2)
            
            # Etiqueta
            label = f"QR: {qr.hash[:8] if qr.hash else 'unknown'}"
            if qr.contrato:
                label += f" [{qr.contrato.get('type', '?')}]"
            
            # Fondo del texto
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_anotado, 
                         (qr.centro[0] - 5, qr.centro[1] - th - 10),
                         (qr.centro[0] + tw + 5, qr.centro[1] - 5),
                         (0, 0, 0), -1)
            cv2.putText(frame_anotado, label, 
                       (qr.centro[0], qr.centro[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_anotado
    
    def decodificar_contrato(self, texto: str) -> Optional[dict]:
        """Decodificar un QR que contiene un contrato.
        
        Args:
            texto: Texto del QR
            
        Returns:
            Contrato parseado o None
        """
        try:
            parsed = json.loads(texto)
            if isinstance(parsed, dict):
                if "type" in parsed:
                    return parsed
                elif "c" in parsed:
                    # Chunked QR
                    return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return None
    
    def verificar_integridad(self, qr: QRDeteccion) -> bool:
        """Verificar la integridad de un QR de contrato.
        
        Args:
            qr: QRDeteccion con contrato
            
        Returns:
            True si el hash coincide
        """
        if not qr.contrato or not qr.hash:
            return False
        
        # Compute hash over contract WITHOUT the hash field
        contrato_sin_hash = {k: v for k, v in qr.contrato.items() if k != "hash"}
        contrato_str = json.dumps(contrato_sin_hash, sort_keys=True, separators=(',', ':'))
        computed_hash = hashlib.sha256(contrato_str.encode()).hexdigest()
        return computed_hash == qr.hash


class ProcesadorQRVideo:
    """Procesador de video para detección continua de QRs.
    
    Integra con ProcesadorVideo de vision-supervision.
    """
    
    def __init__(self, detector_qr: DetectorQR):
        self.detector = detector_qr
        self._callbacks = []
    
    def registrar_callback(self, callback):
        """Registrar callback para cuando se detecta un QR nuevo."""
        self._callbacks.append(callback)
    
    def procesar_frame(self, frame: np.ndarray) -> tuple:
        """Procesar un frame.
        
        Args:
            frame: Imagen BGR
            
        Returns:
            (frame_anotado, lote_qrs)
        """
        lote = self.detector.detectar(frame)
        frame_anotado = self.detector.anotar(frame.copy(), lote)
        
        # Callbacks para QRs nuevos
        for qr in lote.qrs:
            for cb in self._callbacks:
                cb(qr)
        
        return frame_anotado, lote
