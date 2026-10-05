"""
Anotador visual — dibujar detecciones sobre imágenes.
"""
from typing import List, Tuple, Optional
import numpy as np
import cv2
from .deteccion import Deteccion, LoteDetecciones


# Paleta de colores (BGR para OpenCV)
PALETA_COLORES = [
    (255, 0, 0),    # azul
    (0, 255, 0),    # verde
    (0, 0, 255),    # rojo
    (255, 255, 0),  # cian
    (255, 0, 255),  # magenta
    (0, 255, 255),  # amarillo
    (128, 0, 128),  # púrpura
    (255, 165, 0),  # naranja
]


class Anotador:
    """Dibuja bounding boxes, etiquetas y máscaras sobre imágenes."""

    def __init__(
        self,
        grosor_linea: int = 2,
        tamano_fuente: float = 0.5,
        escala_fuente: float = 1.0,
    ):
        self.grosor_linea = grosor_linea
        self.tamano_fuente = tamano_fuente
        self.escala_fuente = escala_fuente

    def _color_clase(self, clase_id: int) -> Tuple[int, int, int]:
        return PALETA_COLORES[clase_id % len(PALETA_COLORES)]

    def anotar(
        self,
        imagen: np.ndarray,
        lote: LoteDetecciones,
        mostrar_confianza: bool = True,
        mostrar_track: bool = True,
    ) -> np.ndarray:
        """Dibujar todas las detecciones sobre una copia de la imagen."""
        salida = imagen.copy()
        for det in lote.detecciones:
            salida = self._anotar_deteccion(salida, det, mostrar_confianza, mostrar_track)
        return salida

    def _anotar_deteccion(
        self,
        imagen: np.ndarray,
        det: Deteccion,
        mostrar_confianza: bool,
        mostrar_track: bool,
    ) -> np.ndarray:
        color = self._color_clase(det.clase_id)
        x1, y1, x2, y2 = det.xyxy.astype(int)

        # Dibujar bounding box
        cv2.rectangle(imagen, (x1, y1), (x2, y2), color, self.grosor_linea)

        # Construir etiqueta
        partes = []
        if det.nombre_clase:
            partes.append(det.nombre_clase)
        if mostrar_confianza:
            partes.append(f"{det.confianza:.2f}")
        if mostrar_track and det.track_id is not None:
            partes.append(f"#{det.track_id}")

        if partes:
            etiqueta = " ".join(partes)
            (tw, th), _ = cv2.getTextSize(
                etiqueta, cv2.FONT_HERSHEY_SIMPLEX, self.tamano_fuente, int(self.escala_fuente)
            )
            # Fondo de etiqueta
            cv2.rectangle(
                imagen,
                (x1, y1 - th - 8),
                (x1 + tw + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                imagen,
                etiqueta,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.tamano_fuente,
                (255, 255, 255),
                int(self.escala_fuente),
                cv2.LINE_AA,
            )

        # Dibujar máscara si existe
        if det.mascara is not None:
            imagen = self._superponer_mascara(imagen, det.mascara, color)

        return imagen

    def _superponer_mascara(
        self,
        imagen: np.ndarray,
        mascara: np.ndarray,
        color: Tuple[int, int, int],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Superponer una máscara semitransparente sobre la imagen."""
        if mascara.shape[:2] != imagen.shape[:2]:
            mascara = cv2.resize(mascara, (imagen.shape[1], imagen.shape[0]))
        mascara_bool = mascara > 0
        overlay = imagen.copy()
        overlay[mascara_bool] = color
        return cv2.addWeighted(overlay, alpha, imagen, 1 - alpha, 0)
