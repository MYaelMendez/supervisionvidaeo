"""
Métricas de evaluación — precisión, recall, IoU, mAP.
"""
from typing import List, Dict, Tuple
import numpy as np
from .deteccion import Deteccion, LoteDetecciones


class Metricas:
    """Métricas de evaluación para detección de objetos."""

    @staticmethod
    def iou_individual(a: Deteccion, b: Deteccion) -> float:
        """IoU entre dos detecciones."""
        return a.iou(b)

    @staticmethod
    def precision_recall(
        predichas: LoteDetecciones,
        ground_truth: LoteDetecciones,
        umbral_iou: float = 0.5,
    ) -> Dict[str, float]:
        """
        Calcular precisión y recall para un frame.
        
        Args:
            predichas: detecciones del modelo
            ground_truth: detecciones reales
            umbral_iou: umbral para considerar un match
            
        Returns:
            dict con 'precision', 'recall', 'tp', 'fp', 'fn'
        """
        tp = 0
        fp = 0
        gt_usadas = set()

        for pred in predichas.detecciones:
            mejor_iou = 0.0
            mejor_gt = None
            for i, gt in enumerate(ground_truth.detecciones):
                if i in gt_usadas:
                    continue
                iou = pred.iou(gt)
                if iou > mejor_iou:
                    mejor_iou = iou
                    mejor_gt = i

            if mejor_iou >= umbral_iou and mejor_gt is not None:
                tp += 1
                gt_usadas.add(mejor_gt)
            else:
                fp += 1

        fn = len(ground_truth.detecciones) - len(gt_usadas)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    @staticmethod
    def map_promedio(
        predichas_por_frame: List[LoteDetecciones],
        gt_por_frame: List[LoteDetecciones],
        umbral_iou: float = 0.5,
    ) -> float:
        """
        Calcular mAP promedio sobre múltiples frames.
        """
        if len(predichas_por_frame) != len(gt_por_frame):
            raise ValueError("El número de frames debe coincidir")

        aps = []
        for pred, gt in zip(predichas_por_frame, gt_por_frame):
            pr = Metricas.precision_recall(pred, gt, umbral_iou)
            # AP simplificado: área bajo la curva precision-recall
            if pr["tp"] + pr["fp"] > 0 and pr["tp"] + pr["fn"] > 0:
                ap = pr["precision"] * pr["recall"]
                aps.append(ap)

        return float(np.mean(aps)) if aps else 0.0
