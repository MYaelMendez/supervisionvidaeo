"""
QR Coder — generación y codificación de códigos QR.

Complementa a qr_vision.py (decodificación) con capacidades de encoding.
Integra con contract_qr.py para generar QRs de contratos inteligentes.
"""

import json
import hashlib
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime, timezone

# QR *encoding* is optional: decoding (qr_vision) needs only OpenCV. Import the
# encoder lazily so `import supervisionvidaeo` works without the qrcode/Pillow
# extras — a missing encoder degrades this module, not the whole package.
try:
    import qrcode
    from qrcode.constants import ERROR_CORRECT_H, ERROR_CORRECT_M, ERROR_CORRECT_L
    from PIL import Image
    _QR_ENCODE_AVAILABLE = True
    _ImageType = Image.Image
except ImportError:  # pragma: no cover - exercised only without the extra
    qrcode = None
    Image = None
    ERROR_CORRECT_H = ERROR_CORRECT_M = ERROR_CORRECT_L = None
    _QR_ENCODE_AVAILABLE = False
    _ImageType = Any  # the annotation is evaluated at class-definition time


def _require_encoder() -> None:
    if not _QR_ENCODE_AVAILABLE:
        raise RuntimeError(
            "QR encoding requires the optional 'qrcode' and 'Pillow' packages. "
            "Install with: pip install 'supervisionvidaeo[qr]'"
        )


@dataclass
class QRCode:
    """Código QR generado."""
    data: str
    image: _ImageType
    version: int
    size: Tuple[int, int]
    filepath: Optional[str] = None
    chunks: int = 1
    chunk_index: int = 0
    total_chunks: int = 1
    hash: Optional[str] = None


@dataclass
class QRLote:
    """Lote de QRs generados (para datos chunked)."""
    qrs: List[QRCode] = field(default_factory=list)
    total_chunks: int = 1
    data_hash: Optional[str] = None


class QRCoder:
    """Generador de códigos QR.
    
    Soporta:
    - Generación simple (un QR para datos pequeños)
    - Generación chunked (múltiples QRs para datos grandes)
    - Integración con contratos inteligentes
    - Diferentes niveles de corrección de errores
    """
    
    # Capacidad aproximada por versión (alphanumeric mode)
    # Versión 40-L: ~4296 chars, Versión 40-M: ~2953 chars
    # These are conservative estimates that account for JSON wrapper overhead
    MAX_CHUNK_SIZES = {
        'L': 3500,
        'M': 2300,
        'Q': 1700,
        'H': 1000,
    }
    
    def __init__(self, error_correction: str = 'H', box_size: int = 10, border: int = 4):
        """
        Args:
            error_correction: L (7%), M (15%), Q (25%), H (30%)
            box_size: Tamaño de cada "box" del QR en pixels
            border: Tamaño del borde en boxes
        """
        self.error_correction = self._parse_error_correction(error_correction)
        self.box_size = box_size
        self.border = border
    
    def _parse_error_correction(self, level: str) -> int:
        """Parse error correction level."""
        levels = {
            'L': ERROR_CORRECT_L,
            'M': ERROR_CORRECT_M,
            'Q': 2,  # ERROR_CORRECT_Q not exported in all versions
            'H': ERROR_CORRECT_H,
        }
        return levels.get(level.upper(), ERROR_CORRECT_H)
    
    def codificar(self, data: str, filepath: Optional[str] = None) -> QRCode:
        """Codificar datos como QR simple.
        
        Args:
            data: Datos a codificar
            filepath: Ruta de salida (opcional)
            
        Returns:
            QRCode generado
        """
        _require_encoder()
        qr = qrcode.QRCode(
            version=None,  # Auto
            error_correction=self.error_correction,
            box_size=self.box_size,
            border=self.border,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        hash_val = hashlib.sha256(data.encode()).hexdigest()[:16]
        
        result = QRCode(
            data=data,
            image=img,
            version=qr.version,
            size=img.size,
            hash=hash_val,
        )
        
        if filepath:
            img.save(filepath)
            result.filepath = filepath
        
        return result
    
    def codificar_chunked(self, data: str, output_dir: Optional[str] = None,
                          prefix: str = "qr_chunk") -> QRLote:
        """Codificar datos grandes como múltiples QRs.
        
        Args:
            data: Datos a codificar
            output_dir: Directorio de salida (opcional)
            prefix: Prefijo para nombres de archivo
            
        Returns:
            QRLote con todos los QRs generados
        """
        max_chunk = self.MAX_CHUNK_SIZES.get(
            {ERROR_CORRECT_L: 'L', ERROR_CORRECT_M: 'M', 2: 'Q', ERROR_CORRECT_H: 'H'}.get(
                self.error_correction, 'H'
            ),
            1663
        )
        
        data_hash = hashlib.sha256(data.encode()).hexdigest()[:16]
        
        if len(data) <= max_chunk:
            # Single QR
            qr = self.codificar(data)
            qr.hash = data_hash
            return QRLote(qrs=[qr], total_chunks=1, data_hash=data_hash)
        
        # Multi-QR
        chunks = [data[i:i+max_chunk] for i in range(0, len(data), max_chunk)]
        qrs = []
        
        for i, chunk in enumerate(chunks):
            chunk_data = json.dumps({
                "c": chunk,
                "n": i + 1,
                "t": len(chunks),
                "h": data_hash,
            })
            
            filepath = None
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                filepath = str(Path(output_dir) / f"{prefix}_{i+1}.png")
            
            qr = self.codificar(chunk_data, filepath)
            qr.chunks = len(chunks)
            qr.chunk_index = i + 1
            qr.total_chunks = len(chunks)
            qr.hash = data_hash
            qrs.append(qr)
        
        return QRLote(qrs=qrs, total_chunks=len(chunks), data_hash=data_hash)
    
    def codificar_contrato(self, contrato: dict, filepath: Optional[str] = None) -> QRCode:
        """Codificar un contrato inteligente como QR.
        
        Args:
            contrato: Diccionario del contrato
            filepath: Ruta de salida (opcional)
            
        Returns:
            QRCode generado
        """
        contrato_str = json.dumps(contrato, ensure_ascii=False, sort_keys=True)
        return self.codificar(contrato_str, filepath)
    
    def codificar_contrato_chunked(self, contrato: dict, output_dir: Optional[str] = None) -> QRLote:
        """Codificar un contrato grande como múltiples QRs.
        
        Args:
            contrato: Diccionario del contrato
            output_dir: Directorio de salida (opcional)
            
        Returns:
            QRLote con todos los QRs generados
        """
        contrato_str = json.dumps(contrato, ensure_ascii=False, sort_keys=True)
        return self.codificar_chunked(contrato_str, output_dir)
    
    def codificar_json(self, data: dict, filepath: Optional[str] = None) -> QRCode:
        """Codificar un diccionario JSON como QR.
        
        Args:
            data: Diccionario a codificar
            filepath: Ruta de salida (opcional)
            
        Returns:
            QRCode generado
        """
        json_str = json.dumps(data, ensure_ascii=False, sort_keys=True)
        return self.codificar(json_str, filepath)
    
    def guardar_lote(self, lote: QRLote, output_dir: str, prefix: str = "qr") -> List[str]:
        """Guardar un lote de QRs a disco.
        
        Args:
            lote: QRLote a guardar
            output_dir: Directorio de salida
            prefix: Prefijo para nombres de archivo
            
        Returns:
            Lista de filepaths generados
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filepaths = []
        
        for qr in lote.qrs:
            if lote.total_chunks == 1:
                filename = f"{prefix}.png"
            else:
                filename = f"{prefix}_{qr.chunk_index}_of_{qr.total_chunks}.png"
            
            filepath = str(Path(output_dir) / filename)
            qr.image.save(filepath)
            qr.filepath = filepath
            filepaths.append(filepath)
        
        return filepaths
    
    def generar_desde_archivo(self, filepath: str, chunked: bool = False,
                               output_dir: Optional[str] = None) -> QRCode | QRLote:
        """Generar QR desde un archivo.
        
        Args:
            filepath: Ruta del archivo
            chunked: Si True, genera múltiples QRs si es necesario
            output_dir: Directorio de salida para QRs
            
        Returns:
            QRCode o QRLote
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.read()
        
        if chunked:
            return self.codificar_chunked(data, output_dir)
        else:
            return self.codificar(data)
    
    def generar_desde_contrato_file(self, contract_filepath: str,
                                     output_dir: Optional[str] = None) -> QRCode | QRLote:
        """Generar QR desde un archivo de contrato JSON.
        
        Args:
            contract_filepath: Ruta del archivo de contrato
            output_dir: Directorio de salida para QRs
            
        Returns:
            QRCode o QRLote
        """
        with open(contract_filepath, 'r', encoding='utf-8') as f:
            contrato = json.load(f)
        
        contrato_str = json.dumps(contrato, ensure_ascii=False, sort_keys=True)
        
        if len(contrato_str) > self.MAX_CHUNK_SIZES.get('H', 1663):
            return self.codificar_chunked(contrato_str, output_dir)
        else:
            filepath = None
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                filepath = str(Path(output_dir) / f"contrato_{contrato.get('hash', 'unknown')[:8]}.png")
            return self.codificar(contrato_str, filepath)


class QRBatchCoder:
    """Generador batch de QRs para múltiples elementos.
    
    Útil para generar QRs de múltiples contratos, keys, o datos en una sola operación.
    """
    
    def __init__(self, coder: Optional[QRCoder] = None):
        self.coder = coder or QRCoder()
    
    def codificar_lista(self, items: List[dict], output_dir: str,
                        prefix: str = "item") -> List[QRCode]:
        """Codificar una lista de diccionarios como QRs.
        
        Args:
            items: Lista de diccionarios
            output_dir: Directorio de salida
            prefix: Prefijo para nombres de archivo
            
        Returns:
            Lista de QRCodes generados
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        qrs = []
        
        for i, item in enumerate(items):
            filepath = str(Path(output_dir) / f"{prefix}_{i+1}.png")
            qr = self.coder.codificar_json(item, filepath)
            qrs.append(qr)
        
        return qrs
    
    def codificar_contratos(self, contratos: List[dict], output_dir: str) -> List[QRCode | QRLote]:
        """Codificar múltiples contratos como QRs.
        
        Args:
            contratos: Lista de contratos
            output_dir: Directorio de salida
            
        Returns:
            Lista de QRCodes o QRLotes
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = []
        
        for contrato in contratos:
            contrato_str = json.dumps(contrato, ensure_ascii=False, sort_keys=True)
            hash_val = contrato.get('hash', 'unknown')[:8]
            
            if len(contrato_str) > 1663:
                result = self.coder.codificar_contrato_chunked(
                    contrato, str(Path(output_dir) / f"contrato_{hash_val}")
                )
            else:
                filepath = str(Path(output_dir) / f"contrato_{hash_val}.png")
                result = self.coder.codificar_contrato(contrato, filepath)
            
            results.append(result)
        
        return results
