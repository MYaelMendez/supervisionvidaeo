"""
Estructuras de datos de detección.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Deteccion:
    """Una detección individual: bounding box + clase + confianza."""
    xyxy: np.ndarray  # [x1, y1, x2, y2]
    clase_id: int
    confianza: float
    nombre_clase: str = ""
    mascara: Optional[np.ndarray] = None
    track_id: Optional[int] = None

    def __post_init__(self):
        self.xyxy = np.asarray(self.xyxy, dtype=np.float32)

    @property
    def area(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0]) * max(0.0, self.xyxy[3] - self.xyxy[1])

    @property
    def centro(self) -> Tuple[float, float]:
        return ((self.xyxy[0] + self.xyxy[2]) / 2, (self.xyxy[1] + self.xyxy[3]) / 2)

    def iou(self, otra: "Deteccion") -> float:
        """Calcular IoU entre dos detecciones."""
        x1 = max(self.xyxy[0], otra.xyxy[0])
        y1 = max(self.xyxy[1], otra.xyxy[1])
        x2 = min(self.xyxy[2], otra.xyxy[2])
        y2 = min(self.xyxy[3], otra.xyxy[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area + otra.area - inter
        return inter / union if union > 0 else 0.0

    def a_dict(self) -> dict:
        return {
            "xyxy": self.xyxy.tolist(),
            "clase_id": self.clase_id,
            "confianza": self.confianza,
            "nombre_clase": self.nombre_clase,
            "track_id": self.track_id,
        }


@dataclass
class LoteDetecciones:
    """Un lote de detecciones para un frame."""
    detecciones: List[Deteccion] = field(default_factory=list)
    frame_id: int = 0

    def __len__(self) -> int:
        return len(self.detecciones)

    def filtrar_por_clase(self, clase_id: int) -> "LoteDetecciones":
        return LoteDetecciones(
            [d for d in self.detecciones if d.clase_id == clase_id],
            self.frame_id,
        )

    def filtrar_por_confianza(self, umbral: float) -> "LoteDetecciones":
        return LoteDetecciones(
            [d for d in self.detecciones if d.confianza >= umbral],
            self.frame_id,
        )

    def a_lista(self) -> List[dict]:
        return [d.a_dict() for d in self.detecciones]
