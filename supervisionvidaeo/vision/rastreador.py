"""
Rastreador de objetos basado en IOU.
Implementación desde principios: asignación greedy por IoU máximo.
"""
from typing import List, Dict, Optional
import numpy as np
from .deteccion import Deteccion, LoteDetecciones


class _Track:
    """Estado interno de un track activo."""
    _siguiente_id = 0

    def __init__(self, deteccion: Deteccion):
        self.track_id = _Track._siguiente_id
        _Track._siguiente_id += 1
        self.deteccion = deteccion
        self.deteccion.track_id = self.track_id
        self.edad = 0
        self.hits = 1
        self.historial: List[Deteccion] = [deteccion]

    def actualizar(self, deteccion: Deteccion):
        """Actualizar el track con una nueva detección."""
        self.deteccion = deteccion
        self.deteccion.track_id = self.track_id
        self.edad = 0
        self.hits += 1
        self.historial.append(deteccion)

    def predecir(self) -> np.ndarray:
        """Predecir próxima posición (constante por ahora)."""
        return self.deteccion.xyxy.copy()

    def envejecer(self):
        """Incrementar edad sin detección."""
        self.edad += 1


class RastreadorIOU:
    """
    Rastreador de objetos basado en IOU.
    
    Asigna detecciones a tracks existentes usando IoU máximo.
    Los tracks sin coincidencia durante `max_edad` frames se eliminan.
    """

    def __init__(self, umbral_iou: float = 0.3, max_edad: int = 5):
        self.umbral_iou = umbral_iou
        self.max_edad = max_edad
        self.tracks: Dict[int, _Track] = {}

    def reiniciar(self):
        """Reiniciar todos los tracks."""
        self.tracks.clear()
        _Track._siguiente_id = 0

    def actualizar(self, lote: LoteDetecciones) -> LoteDetecciones:
        """
        Actualizar tracks con un nuevo lote de detecciones.
        Retorna un LoteDetecciones con track_ids asignados.
        """
        # 1. Envejecer tracks existentes
        for track in self.tracks.values():
            track.envejecer()

        # 2. Asignar detecciones a tracks (greedy por IoU máximo)
        asignados = set()
        detecciones_ordenadas = sorted(
            lote.detecciones, key=lambda d: d.confianza, reverse=True
        )

        for det in detecciones_ordenadas:
            mejor_track = None
            mejor_iou = self.umbral_iou

            for track in self.tracks.values():
                if track.track_id in asignados:
                    continue
                iou = det.iou(track.deteccion)
                if iou > mejor_iou:
                    mejor_iou = iou
                    mejor_track = track

            if mejor_track is not None:
                mejor_track.actualizar(det)
                asignados.add(mejor_track.track_id)
            else:
                # Crear nuevo track
                nuevo = _Track(det)
                self.tracks[nuevo.track_id] = nuevo
                asignados.add(nuevo.track_id)

        # 3. Eliminar tracks viejos
        tracks_a_eliminar = [
            tid for tid, t in self.tracks.items() if t.edad > self.max_edad
        ]
        for tid in tracks_a_eliminar:
            del self.tracks[tid]

        # 4. Retornar detecciones con track_ids
        resultado = []
        for track in self.tracks.values():
            if track.edad == 0:  # solo tracks actualizados este frame
                resultado.append(track.deteccion)

        return LoteDetecciones(resultado, lote.frame_id)
