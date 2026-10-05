"""Tests para QR Vision — detección y decodificación de códigos QR."""

import pytest
import numpy as np
import cv2
import json
import tempfile
from pathlib import Path

from supervisionvidaeo.vision.qr_vision import (
    QRDeteccion,
    LoteQRs,
    DetectorQR,
    ProcesadorQRVideo,
)


class TestQRDeteccion:
    """Tests para la clase QRDeteccion."""

    def test_creacion_basica(self):
        puntos = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
        qr = QRDeteccion(
            texto="test",
            puntos=puntos,
            centro=(30, 30),
            confianza=0.9,
        )
        assert qr.texto == "test"
        assert qr.confianza == 0.9
        assert qr.centro == (30, 30)

    def test_con_contrato(self):
        contrato = {
            "type": "escrow",
            "terms": {"amount": 100},
            "parties": ["alice", "bob"],
            "hash": "abc123",
        }
        puntos = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], dtype=np.float32)
        qr = QRDeteccion(
            texto=json.dumps(contrato),
            puntos=puntos,
            centro=(20, 20),
            contrato=contrato,
            hash="abc123",
        )
        assert qr.contrato is not None
        assert qr.contrato["type"] == "escrow"
        assert qr.hash == "abc123"


class TestLoteQRs:
    """Tests para la clase LoteQRs."""

    def test_lote_vacio(self):
        lote = LoteQRs()
        assert len(lote.qrs) == 0
        assert lote.frame_id == 0

    def test_lote_con_qrs(self):
        lote = LoteQRs(frame_id=1)
        puntos = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        qr = QRDeteccion(texto="a", puntos=puntos, centro=(5, 5))
        lote.qrs.append(qr)
        assert len(lote.qrs) == 1
        assert lote.frame_id == 1


class TestDetectorQR:
    """Tests para la clase DetectorQR."""

    def test_deteccion_vacia(self):
        """Detector en frame sin QRs no debe detectar nada."""
        detector = DetectorQR(usar_tracking=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lote = detector.detectar(frame)
        assert len(lote.qrs) == 0

    def test_deteccion_con_qr_real(self):
        """Detector debe detectar un QR generado con OpenCV."""
        detector = DetectorQR(usar_tracking=False)
        
        # Generar un QR real con mejor contraste y tamaño
        import qrcode
        texto = '{"type":"escrow","hash":"test123"}'
        qr_img = qrcode.make(texto)
        qr_array = np.array(qr_img.convert("RGB"))
        qr_bgr = cv2.cvtColor(qr_array, cv2.COLOR_RGB2BGR)
        
        # Insertar QR en el frame con fondo blanco para mejor contraste
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        h, w = qr_bgr.shape[:2]
        y, x = 100, 100
        frame[y:y+h, x:x+w] = qr_bgr
        
        lote = detector.detectar(frame)
        # Debe detectar al menos el QR
        assert len(lote.qrs) >= 1

    def test_anotar_frame(self):
        """Anotar no debe modificar el frame original."""
        detector = DetectorQR(usar_tracking=False)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        frame_copy = frame.copy()
        
        lote = LoteQRs(frame_id=1)
        puntos = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
        qr = QRDeteccion(texto="test", puntos=puntos, centro=(150, 150))
        lote.qrs.append(qr)
        
        anotado = detector.anotar(frame, lote)
        # El frame original no debe cambiar
        np.testing.assert_array_equal(frame, frame_copy)
        # El anotado debe ser diferente (se dibujó el contorno)
        assert not np.array_equal(frame, anotado)

    def test_decodificar_contrato_valido(self):
        """Decodificar un QR con contrato JSON válido."""
        detector = DetectorQR(usar_tracking=False)
        contrato = {"type": "escrow", "terms": {}, "parties": ["a"], "hash": "abc"}
        texto = json.dumps(contrato)
        resultado = detector.decodificar_contrato(texto)
        assert resultado is not None
        assert resultado["type"] == "escrow"

    def test_decodificar_contrato_invalido(self):
        """Decodificar un QR que no es contrato."""
        detector = DetectorQR(usar_tracking=False)
        resultado = detector.decodificar_contrato("hola mundo")
        assert resultado is None

    def test_verificar_integridad_valida(self):
        """Verificar integridad de un QR con hash correcto."""
        detector = DetectorQR(usar_tracking=False)
        # Contract without hash field
        contrato_sin_hash = {"type": "escrow", "terms": {}, "parties": ["a"]}
        contrato_str = json.dumps(contrato_sin_hash, sort_keys=True, separators=(",", ":"))
        import hashlib
        hash_real = hashlib.sha256(contrato_str.encode()).hexdigest()
        # Add hash to contract
        contrato = {**contrato_sin_hash, "hash": hash_real}
        
        puntos = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        qr = QRDeteccion(
            texto=json.dumps(contrato),
            puntos=puntos,
            centro=(5, 5),
            contrato=contrato,
            hash=hash_real,
        )
        assert detector.verificar_integridad(qr) is True

    def test_verificar_integridad_invalida(self):
        """Verificar integridad de un QR con hash incorrecto."""
        detector = DetectorQR(usar_tracking=False)
        contrato = {"type": "escrow", "terms": {}, "parties": ["a"], "hash": "wrong"}
        
        puntos = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        qr = QRDeteccion(
            texto=json.dumps(contrato),
            puntos=puntos,
            centro=(5, 5),
            contrato=contrato,
            hash="wrong",
        )
        assert detector.verificar_integridad(qr) is False


class TestProcesadorQRVideo:
    """Tests para la clase ProcesadorQRVideo."""

    def test_procesar_frame_vacio(self):
        """Procesar un frame sin QRs."""
        detector = DetectorQR(usar_tracking=False)
        procesador = ProcesadorQRVideo(detector)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        anotado, lote = procesador.procesar_frame(frame)
        assert len(lote.qrs) == 0
        assert anotado.shape == frame.shape

    def test_callback_ejecutado(self):
        """Callback debe ejecutarse cuando se detecta un QR."""
        detector = DetectorQR(usar_tracking=False)
        procesador = ProcesadorQRVideo(detector)
        
        detectados = []
        procesador.registrar_callback(lambda qr: detectados.append(qr))
        
        # Generar frame con QR real
        import qrcode
        texto = '{"type":"test","hash":"abc"}'
        qr_img = qrcode.make(texto)
        qr_array = np.array(qr_img.convert("RGB"))
        qr_bgr = cv2.cvtColor(qr_array, cv2.COLOR_RGB2BGR)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = qr_bgr.shape[:2]
        frame[100:100+h, 100:100+w] = qr_bgr
        
        procesador.procesar_frame(frame)
        assert len(detectados) >= 1
