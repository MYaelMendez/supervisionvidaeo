"""
Procesador de video — pipeline frame a frame.
"""
from typing import Callable, Optional, Iterator, Tuple
import numpy as np
import cv2
from .deteccion import LoteDetecciones


class ProcesadorVideo:
    """
    Procesa video frame a frame con un pipeline de detección + rastreo + anotación.
    
    Uso:
        procesador = ProcesadorVideo(detector, rastreador, anotador)
        for frame_anotado in procesador.procesar("video.mp4"):
            # mostrar o guardar
    """

    def __init__(
        self,
        detector: Callable[[np.ndarray], LoteDetecciones],
        rastreador=None,
        anotador=None,
        max_frames: Optional[int] = None,
    ):
        self.detector = detector
        self.rastreador = rastreador
        self.anotador = anotador
        self.max_frames = max_frames

    def procesar(
        self,
        fuente: str,
        mostrar: bool = False,
        guardar: Optional[str] = None,
    ) -> Iterator[np.ndarray]:
        """
        Procesar un archivo de video o cámara.
        
        Args:
            fuente: ruta de video o índice de cámara (0, 1, ...)
            mostrar: mostrar ventana en tiempo real
            guardar: ruta de salida para guardar video anotado
            
        Yields:
            Frames anotados como np.ndarray
        """
        cap = cv2.VideoCapture(fuente)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir la fuente: {fuente}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if guardar:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(guardar, fourcc, fps, (ancho, alto))

        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.max_frames and frame_id >= self.max_frames:
                    break

                # Detectar
                lote = self.detector(frame)
                lote.frame_id = frame_id

                # Rastrear
                if self.rastreador:
                    lote = self.rastreador.actualizar(lote)

                # Anotar
                if self.anotador:
                    frame = self.anotador.anotar(frame, lote)

                # Guardar
                if writer:
                    writer.write(frame)

                # Mostrar
                if mostrar:
                    cv2.imshow("Vision Supervision", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                yield frame
                frame_id += 1
        finally:
            cap.release()
            if writer:
                writer.release()
            if mostrar:
                cv2.destroyAllWindows()

    def procesar_imagen(self, imagen: np.ndarray) -> Tuple[np.ndarray, LoteDetecciones]:
        """Procesar una sola imagen."""
        lote = self.detector(imagen)
        lote.frame_id = 0
        if self.rastreador:
            lote = self.rastreador.actualizar(lote)
        if self.anotador:
            imagen = self.anotador.anotar(imagen, lote)
        return imagen, lote
