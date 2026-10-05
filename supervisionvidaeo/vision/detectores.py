"""
Detector simple basado en color — sin modelos ML.
Útil para demos y pruebas sin dependencias externas.
"""
from typing import List, Tuple
import numpy as np
import cv2
from .deteccion import Deteccion, LoteDetecciones


class DetectorColor:
    """
    Detecta objetos por color en espacio HSV.
    
    No requiere modelos ML — usa segmentación por color.
    Ideal para demos, pruebas y casos de uso simples.
    """

    # Colores predefinidos en HSV (H: 0-179, S: 0-255, V: 0-255)
    COLORES = {
        "rojo": ([0, 100, 100], [10, 255, 255]),
        "rojo2": ([160, 100, 100], [179, 255, 255]),  # rojo envuelve en HSV
        "verde": ([40, 50, 50], [80, 255, 255]),
        "azul": ([100, 50, 50], [130, 255, 255]),
        "amarillo": ([20, 100, 100], [35, 255, 255]),
        "naranja": ([10, 100, 100], [20, 255, 255]),
        "morado": ([130, 50, 50], [160, 255, 255]),
        "blanco": ([0, 0, 200], [179, 30, 255]),
        "negro": ([0, 0, 0], [179, 255, 50]),
    }

    def __init__(
        self,
        colores: List[str] = None,
        area_minima: int = 500,
        umbral_confianza: float = 0.3,
    ):
        self.colores = colores or ["rojo", "verde", "azul"]
        self.area_minima = area_minima
        self.umbral_confianza = umbral_confianza

    def detectar(self, imagen: np.ndarray) -> LoteDetecciones:
        """
        Detectar objetos por color en la imagen.
        
        Args:
            imagen: imagen BGR (formato OpenCV)
            
        Returns:
            LoteDetecciones con las detecciones encontradas
        """
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        detecciones = []
        clase_id = 0

        for nombre_color in self.colores:
            if nombre_color not in self.COLORES:
                continue

            # Manejar rojo que envuelve en HSV
            if nombre_color == "rojo":
                lower1 = np.array(self.COLORES["rojo"][0])
                upper1 = np.array(self.COLORES["rojo"][1])
                lower2 = np.array(self.COLORES["rojo2"][0])
                upper2 = np.array(self.COLORES["rojo2"][1])
                mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
            elif nombre_color == "rojo2":
                continue  # ya se procesó con "rojo"
            else:
                lower = np.array(self.COLORES[nombre_color][0])
                upper = np.array(self.COLORES[nombre_color][1])
                mask = cv2.inRange(hsv, lower, upper)

            # Encontrar contornos
            contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contornos:
                area = cv2.contourArea(cnt)
                if area < self.area_minima:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                # Confianza basada en la relación de área del contorno vs bounding box
                confianza = min(1.0, area / (w * h + 1)) if w * h > 0 else 0.0

                if confianza >= self.umbral_confianza:
                    det = Deteccion(
                        xyxy=np.array([x, y, x + w, y + h], dtype=np.float32),
                        clase_id=clase_id,
                        confianza=confianza,
                        nombre_clase=nombre_color,
                    )
                    detecciones.append(det)

            clase_id += 1

        return LoteDetecciones(detecciones)


class DetectorMovimiento:
    """
    Detecta movimiento entre frames consecutivos.
    Usa diferencia de frames + umbralización.
    """

    def __init__(self, umbral: int = 25, area_minima: int = 300):
        self.umbral = umbral
        self.area_minima = area_minima
        self.frame_anterior = None

    def detectar(self, imagen: np.ndarray) -> LoteDetecciones:
        """
        Detectar movimiento comparando con el frame anterior.
        """
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (21, 21), 0)

        if self.frame_anterior is None:
            self.frame_anterior = gris
            return LoteDetecciones([])

        # Diferencia absoluta
        diff = cv2.absdiff(self.frame_anterior, gris)
        self.frame_anterior = gris.copy()

        # Umbralizar
        _, thresh = cv2.threshold(diff, self.umbral, 255, cv2.THRESH_BINARY)

        # Dilatar para rellenar huecos
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Encontrar contornos
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detecciones = []
        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area < self.area_minima:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            det = Deteccion(
                xyxy=np.array([x, y, x + w, y + h], dtype=np.float32),
                clase_id=0,
                confianza=min(1.0, area / 10000),
                nombre_clase="movimiento",
            )
            detecciones.append(det)

        return LoteDetecciones(detecciones)

    def reiniciar(self):
        """Reiniciar el frame anterior."""
        self.frame_anterior = None
