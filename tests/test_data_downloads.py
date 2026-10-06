"""Resume/md5/unzip logic of download_bradd.py and WFS paging/class check of download_deter_prodes.py."""
from __future__ import annotations

import hashlib
import io
import json
import threading
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import pytest
import requests

from scripts import download_bradd as dl
from scripts import download_deter_prodes as tb


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("BraDD/meta.csv", ",alert_idx\n0,0\n")
        archive.writestr("BraDD/Samples/0000000_2020-08-01.pt", b"x" * 5000)
    return buf.getvalue()


PAYLOAD = _zip_bytes()


def _handler(honour_range: bool, seen: list):
    class Handler(BaseHTTPRequestHandler):
        """Minimal GET handler with optional HTTP Range support."""
        def do_GET(self) -> None:  # noqa: N802
            """Serve PAYLOAD from the requested byte offset."""
            rng = self.headers.get("Range")
            seen.append(rng)
            start = int(rng.split("=")[1].rstrip("-")) if (rng and honour_range) else 0
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(PAYLOAD) - start))
            self.end_headers()
            self.wfile.write(PAYLOAD[start:])

        def log_message(self, *args) -> None:
            """Silence request logging."""
            return None
    return Handler


@pytest.fixture(params=[True, False], ids=["range", "no-range"])
def server(request) -> Iterator[tuple[str, list, bool]]:
    """Local HTTP server serving PAYLOAD, with or without Range support."""
    seen: list = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(request.param, seen))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/f.zip", seen, request.param
    httpd.shutdown()


def _cfg(url, dest, md5, delete=False, min_free=0.0):
    return {"url": url, "md5": md5, "dest_dir": str(dest), "zip_name": "b.zip", "chunk_mb": 0.001,
            "timeout_s": 10, "delete_zip_after_unzip": delete, "min_free_gb": min_free}


def test_resume_from_partial_file(server, tmp_path) -> None:
    """A partial .part file is resumed (or restarted) and the md5 covers the whole file."""
    url, seen, _ = server
    part = tmp_path / "b.zip.part"
    part.write_bytes(PAYLOAD[:1234])
    digest = dl.download_resumable(requests.Session(), url, part, 512, 10)
    assert seen == ["bytes=1234-"]
    assert part.read_bytes() == PAYLOAD and digest == hashlib.md5(PAYLOAD).hexdigest()


def test_complete_part_gets_416(server, tmp_path) -> None:
    """A complete .part file is accepted when the server answers 416."""
    url = server[0]
    part = tmp_path / "b.zip.part"
    part.write_bytes(PAYLOAD)
    assert dl.download_resumable(requests.Session(), url, part, 512, 10) == hashlib.md5(PAYLOAD).hexdigest()


def test_fetch_verify_unzip_and_delete(server, tmp_path) -> None:
    """Full flow unzips and deletes the zip when free space is under the threshold."""
    url = server[0]
    dest = dl.fetch_verify_unzip(_cfg(url, tmp_path, hashlib.md5(PAYLOAD).hexdigest(), min_free=1e9),
                                 requests.Session())
    assert (dest / "BraDD" / "meta.csv").exists() and not (tmp_path / "b.zip").exists()


def test_md5_mismatch_keeps_file(server, tmp_path) -> None:
    """An md5 mismatch raises, keeps the download and does not unzip."""
    with pytest.raises(ValueError, match="md5 mismatch"):
        dl.fetch_verify_unzip(_cfg(server[0], tmp_path, "0" * 32), requests.Session())
    assert (tmp_path / "b.zip.part").exists() and not (tmp_path / "BraDD").exists()


def test_unsafe_zip_rejected(tmp_path) -> None:
    """Archive members escaping the destination are refused."""
    path = tmp_path / "evil.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        dl.safe_unzip(path, tmp_path / "out")


def _fake_pages(sizes):
    pages = [{"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"classname": cls, "gid": i}, "geometry": None}
        for i, cls in enumerate(["DESMATAMENTO_CR", "MINERACAO", "CICATRIZ_DE_QUEIMADA", "NEW_CLASS"] * 3)][:n]}
        for n in sizes]
    responses = []
    for page in pages:
        response = mock.Mock()
        response.json.return_value = page
        responses.append(response)
    session = mock.Mock()
    session.get.side_effect = responses
    return session


def test_wfs_pagination_and_resume(tmp_path) -> None:
    """WFS pages advance startIndex by count, stop on a short page and are cached."""
    layer = {"type_name": "deter-amz:deter_amz", "sort_by": "gid", "cql_filter": "view_date >= '2020-01-01'"}
    session = _fake_pages([4, 4, 2])
    total = tb.fetch_wfs_pages(session, "http://wfs", layer, "2.0.0", "application/json", 4, 5, tmp_path / "p")
    starts = [c.kwargs["params"]["startIndex"] for c in session.get.call_args_list]
    assert total == 10 and starts == [0, 4, 8]
    assert session.get.call_args_list[0].kwargs["params"]["CQL_FILTER"] == layer["cql_filter"]
    again = mock.Mock()
    assert tb.fetch_wfs_pages(again, "http://wfs", layer, "2.0.0", "application/json", 4, 5, tmp_path / "p") == 10
    again.get.assert_not_called()
    assert tb.merge_pages(tmp_path / "p", tmp_path / "all.geojson") == 10


def test_verify_deter_classes(tmp_path) -> None:
    """Class check reports found, missing and unexpected classname values."""
    session = _fake_pages([4])
    tb.fetch_wfs_pages(session, "http://wfs", {"type_name": "x"}, "2.0.0", "application/json", 10, 5, tmp_path)
    tb.merge_pages(tmp_path, tmp_path / "deter.geojson")
    report = tb.verify_deter_classes(tmp_path / "deter.geojson")
    assert report["found"] == ["CICATRIZ_DE_QUEIMADA", "DESMATAMENTO_CR", "MINERACAO", "NEW_CLASS"]
    assert report["unexpected"] == ["NEW_CLASS"] and "DESMATAMENTO_VEG" in report["missing"]
    assert json.loads((tmp_path / "deter.geojson").read_text())["type"] == "FeatureCollection"
