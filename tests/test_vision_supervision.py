"""
Tests para Vision Supervision.
"""
import numpy as np
import cv2
import pytest
from supervisionvidaeo.vision import (
    Deteccion,
    LoteDetecciones,
    Anotador,
    RastreadorIOU,
    DetectorColor,
    DetectorMovimiento,
    Metricas,
)


class TestDeteccion:
    def test_creacion_basica(self):
        det = Deteccion(
            xyxy=np.array([10, 20, 50, 80]),
            clase_id=0,
            confianza=0.95,
            nombre_clase="objeto",
        )
        assert det.clase_id == 0
        assert det.confianza == 0.95
        assert det.nombre_clase == "objeto"

    def test_area(self):
        det = Deteccion(
            xyxy=np.array([0, 0, 10, 20]),
            clase_id=0,
            confianza=1.0,
        )
        assert det.area == 200.0

    def test_centro(self):
        det = Deteccion(
            xyxy=np.array([10, 20, 30, 60]),
            clase_id=0,
            confianza=1.0,
        )
        assert det.centro == (20.0, 40.0)

    def test_iou_identico(self):
        det1 = Deteccion(xyxy=np.array([0, 0, 10, 10]), clase_id=0, confianza=1.0)
        det2 = Deteccion(xyxy=np.array([0, 0, 10, 10]), clase_id=0, confianza=1.0)
        assert det1.iou(det2) == 1.0

    def test_iou_sin_solapamiento(self):
        det1 = Deteccion(xyxy=np.array([0, 0, 10, 10]), clase_id=0, confianza=1.0)
        det2 = Deteccion(xyxy=np.array([20, 20, 30, 30]), clase_id=0, confianza=1.0)
        assert det1.iou(det2) == 0.0

    def test_iou_parcial(self):
        det1 = Deteccion(xyxy=np.array([0, 0, 10, 10]), clase_id=0, confianza=1.0)
        det2 = Deteccion(xyxy=np.array([5, 5, 15, 15]), clase_id=0, confianza=1.0)
        assert 0.0 < det1.iou(det2) < 1.0

    def test_a_dict(self):
        det = Deteccion(
            xyxy=np.array([1, 2, 3, 4]),
            clase_id=1,
            confianza=0.8,
            nombre_clase="test",
            track_id=5,
        )
        d = det.a_dict()
        assert d["clase_id"] == 1
        assert d["confianza"] == 0.8
        assert d["track_id"] == 5


class TestLoteDetecciones:
    def test_filtrar_por_clase(self):
        lote = LoteDetecciones([
            Deteccion(xyxy=np.array([0, 0, 10, 10]), clase_id=0, confianza=0.9),
            Deteccion(xyxy=np.array([20, 20, 30, 30]), clase_id=1, confianza=0.8),
            Deteccion(xyxy=np.array([40, 40, 50, 50]), clase_id=0, confianza=0.7),
        ])
        filtrado = lote.filtrar_por_clase(0)
        assert len(filtrado) == 2

    def test_filtrar_por_confianza(self):
        lote = LoteDetecciones([
            Deteccion(xyxy=np.array([0, 0, 10, 10]), clase_id=0, confianza=0.9),
            Deteccion(xyxy=np.array([20, 20, 30, 30]), clase_id=0, confianza=0.5),
            Deteccion(xyxy=np.array([40, 40, 50, 50]), clase_id=0, confianza=0.3),
        ])
        filtrado = lote.filtrar_por_confianza(0.6)
        assert len(filtrado) == 1


class TestAnotador:
    def test_anotar_no_modifica_original(self):
        imagen = np.zeros((100, 100, 3), dtype=np.uint8)
        original = imagen.copy()
        lote = LoteDetecciones([
            Deteccion(xyxy=np.array([10, 10, 50, 50]), clase_id=0, confianza=0.9, nombre_clase="test"),
        ])
        anotador = Anotador()
        resultado = anotador.anotar(imagen, lote)
        assert np.array_equal(imagen, original)
        assert not np.array_equal(resultado, original)


class TestRastreadorIOU:
    def test_rastreo_basico(self):
        rastreador = RastreadorIOU(umbral_iou=0.3, max_edad=2)
        
        # Frame 1: un objeto
        lote1 = LoteDetecciones([
            Deteccion(xyxy=np.array([10, 10, 50, 50]), clase_id=0, confianza=0.9),
        ], frame_id=0)
        r1 = rastreador.actualizar(lote1)
        assert len(r1) == 1
        assert r1.detecciones[0].track_id is not None

        # Frame 2: mismo objeto movido
        lote2 = LoteDetecciones([
            Deteccion(xyxy=np.array([12, 12, 52, 52]), clase_id=0, confianza=0.9),
        ], frame_id=1)
        r2 = rastreador.actualizar(lote2)
        assert len(r2) == 1
        assert r2.detecciones[0].track_id == r1.detecciones[0].track_id

    def test_reiniciar(self):
        rastreador = RastreadorIOU()
        lote = LoteDetecciones([
            Deteccion(xyxy=np.array([10, 10, 50, 50]), clase_id=0, confianza=0.9),
        ])
        rastreador.actualizar(lote)
        assert len(rastreador.tracks) == 1
        rastreador.reiniciar()
        assert len(rastreador.tracks) == 0


class TestDetectorColor:
    def test_detectar_rojo(self):
        # Crear imagen con cuadrado rojo
        imagen = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(imagen, (20, 20), (60, 60), (0, 0, 255), -1)  # rojo en BGR
        
        detector = DetectorColor(colores=["rojo"], area_minima=100)
        lote = detector.detectar(imagen)
        assert len(lote) >= 1
        assert lote.detecciones[0].nombre_clase == "rojo"

    def test_sin_detecciones(self):
        imagen = np.zeros((100, 100, 3), dtype=np.uint8)
        detector = DetectorColor(colores=["rojo"], area_minima=100)
        lote = detector.detectar(imagen)
        assert len(lote) == 0


class TestDetectorMovimiento:
    def test_sin_movimiento_inicial(self):
        imagen = np.zeros((100, 100, 3), dtype=np.uint8)
        detector = DetectorMovimiento()
        lote = detector.detectar(imagen)
        assert len(lote) == 0

    def test_detecta_movimiento(self):
        detector = DetectorMovimiento()
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        detector.detectar(frame1)
        
        frame2 = frame1.copy()
        cv2.rectangle(frame2, (20, 20), (60, 60), (255, 255, 255), -1)
        lote = detector.detectar(frame2)
        assert len(lote) >= 1


class TestMetricas:
    def test_precision_recall_perfecto(self):
        gt = LoteDetecciones([
            Deteccion(xyxy=np.array([10, 10, 50, 50]), clase_id=0, confianza=1.0),
        ])
        pred = LoteDetecciones([
            Deteccion(xyxy=np.array([10, 10, 50, 50]), clase_id=0, confianza=0.9),
        ])
        pr = Metricas.precision_recall(pred, gt, umbral_iou=0.5)
        assert pr["precision"] == 1.0
        assert pr["recall"] == 1.0
        assert pr["tp"] == 1
        assert pr["fp"] == 0
        assert pr["fn"] == 0

    def test_precision_recall_fallo(self):
        gt = LoteDetecciones([
            Deteccion(xyxy=np.array([10, 10, 50, 50]), clase_id=0, confianza=1.0),
        ])
        pred = LoteDetecciones([
            Deteccion(xyxy=np.array([100, 100, 150, 150]), clase_id=0, confianza=0.9),
        ])
        pr = Metricas.precision_recall(pred, gt, umbral_iou=0.5)
        assert pr["precision"] == 0.0
        assert pr["recall"] == 0.0
        assert pr["tp"] == 0
        assert pr["fp"] == 1
        assert pr["fn"] == 1
