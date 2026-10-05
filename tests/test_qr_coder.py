"""Tests para QR Coder — generación y codificación de códigos QR."""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np

pytest.importorskip("qrcode")
pytest.importorskip("PIL")

from supervisionvidaeo.vision.qr_coder import QRCode, QRLote, QRCoder, QRBatchCoder


class TestQRCode:
    """Tests para la clase QRCode."""

    def test_creacion_basica(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), "white")
        qr = QRCode(
            data="test",
            image=img,
            version=1,
            size=(100, 100),
            hash="abc123",
        )
        assert qr.data == "test"
        assert qr.version == 1
        assert qr.size == (100, 100)
        assert qr.hash == "abc123"
        assert qr.chunks == 1

    def test_con_chunks(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), "white")
        qr = QRCode(
            data="chunk data",
            image=img,
            version=5,
            size=(200, 200),
            chunks=3,
            chunk_index=2,
            total_chunks=3,
            hash="def456",
        )
        assert qr.chunks == 3
        assert qr.chunk_index == 2
        assert qr.total_chunks == 3


class TestQRLote:
    """Tests para la clase QRLote."""

    def test_lote_vacio(self):
        lote = QRLote()
        assert len(lote.qrs) == 0
        assert lote.total_chunks == 1

    def test_lote_con_qrs(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), "white")
        qr = QRCode(data="a", image=img, version=1, size=(100, 100))
        lote = QRLote(qrs=[qr], total_chunks=1, data_hash="abc")
        assert len(lote.qrs) == 1
        assert lote.data_hash == "abc"


class TestQRCoder:
    """Tests para la clase QRCoder."""

    def test_codificar_simple(self):
        coder = QRCoder()
        qr = coder.codificar("hello world")
        assert qr.data == "hello world"
        assert qr.version >= 1
        assert qr.size[0] > 0
        assert qr.hash is not None

    def test_codificar_con_filepath(self):
        coder = QRCoder()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "test_qr.png")
            qr = coder.codificar("test data", filepath)
            assert qr.filepath == filepath
            assert Path(filepath).exists()

    def test_codificar_json(self):
        coder = QRCoder()
        data = {"type": "escrow", "amount": 100}
        qr = coder.codificar_json(data)
        assert qr.data is not None
        parsed = json.loads(qr.data)
        assert parsed["type"] == "escrow"

    def test_codificar_contrato(self):
        coder = QRCoder()
        contrato = {
            "type": "escrow",
            "terms": {"amount": 100},
            "parties": ["alice", "bob"],
            "hash": "abc123",
        }
        qr = coder.codificar_contrato(contrato)
        assert qr.data is not None
        parsed = json.loads(qr.data)
        assert parsed["type"] == "escrow"

    def test_codificar_chunked_pequeño(self):
        coder = QRCoder()
        # Datos pequeños deben generar un solo QR
        lote = coder.codificar_chunked("small data")
        assert lote.total_chunks == 1
        assert len(lote.qrs) == 1

    def test_codificar_chunked_grande(self):
        coder = QRCoder()
        # Datos grandes deben generar múltiples QRs
        big_data = "x" * 5000
        lote = coder.codificar_chunked(big_data)
        assert lote.total_chunks > 1
        assert len(lote.qrs) == lote.total_chunks

    def test_guardar_lote(self):
        coder = QRCoder()
        with tempfile.TemporaryDirectory() as tmpdir:
            lote = coder.codificar_chunked("test data for saving")
            filepaths = coder.guardar_lote(lote, tmpdir, "test")
            assert len(filepaths) == len(lote.qrs)
            for fp in filepaths:
                assert Path(fp).exists()

    def test_generar_desde_archivo(self):
        coder = QRCoder()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Crear archivo de prueba
            filepath = Path(tmpdir) / "test.txt"
            filepath.write_text("hello from file", encoding="utf-8")
            
            qr = coder.generar_desde_archivo(str(filepath))
            assert qr.data == "hello from file"

    def test_generar_desde_contrato_file(self):
        coder = QRCoder()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Crear archivo de contrato
            contrato = {
                "type": "escrow",
                "terms": {"amount": 50},
                "parties": ["alice"],
                "hash": "test123",
            }
            filepath = Path(tmpdir) / "contract.json"
            filepath.write_text(json.dumps(contrato), encoding="utf-8")
            
            qr = coder.generar_desde_contrato_file(str(filepath))
            assert qr.data is not None
            parsed = json.loads(qr.data)
            assert parsed["type"] == "escrow"

    def test_niveles_error_correction(self):
        """Probar diferentes niveles de corrección de errores."""
        for level in ['L', 'M', 'H']:
            coder = QRCoder(error_correction=level)
            qr = coder.codificar("test")
            assert qr.version >= 1


class TestQRBatchCoder:
    """Tests para la clase QRBatchCoder."""

    def test_codificar_lista(self):
        coder = QRCoder()
        batch = QRBatchCoder(coder)
        with tempfile.TemporaryDirectory() as tmpdir:
            items = [
                {"name": "item1", "value": 1},
                {"name": "item2", "value": 2},
                {"name": "item3", "value": 3},
            ]
            qrs = batch.codificar_lista(items, tmpdir, "item")
            assert len(qrs) == 3
            for qr in qrs:
                assert qr.filepath is not None
                assert Path(qr.filepath).exists()

    def test_codificar_contratos(self):
        coder = QRCoder()
        batch = QRBatchCoder(coder)
        with tempfile.TemporaryDirectory() as tmpdir:
            contratos = [
                {"type": "escrow", "hash": "abc", "terms": {}},
                {"type": "multisig", "hash": "def", "terms": {}},
            ]
            results = batch.codificar_contratos(contratos, tmpdir)
            assert len(results) == 2
