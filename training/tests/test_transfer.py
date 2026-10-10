"""Chunked, resumable transfers in the group worker, against a fake coordinator that drops connections on purpose."""

import http.server
import json
import threading
import types
import unittest
from urllib.parse import parse_qs, urlparse

import torch
from safetensors.torch import save as st_save

from training import group_worker


class FakeCoordinator:
    """Serves a body with byte ranges, drops some requests partway through, and records uploaded delta parts."""

    def __init__(self, body: bytes, drop_requests: set[int] = frozenset()):
        self.body = body
        self.drop = set(drop_requests)
        self.requests = 0
        self.parts: dict[int, bytes] = {}
        self.part_count = None
        self.upload_drops = set()
        self.uploads = 0
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.requests += 1
                rng = self.headers.get("Range")
                start, end = (int(x) for x in rng.split("=", 1)[1].split("-"))
                end = min(end, len(outer.body) - 1)
                piece = outer.body[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(piece)))
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(outer.body)}")
                self.send_header("x-pw-round", "1")
                self.send_header("x-pw-sha256", outer.sha)
                self.end_headers()
                if outer.requests in outer.drop:
                    self.wfile.write(piece[: len(piece) // 2])  # the connection drops mid-body
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(piece)

            def do_POST(self):
                outer.uploads += 1
                q = parse_qs(urlparse(self.path).query)
                length = int(self.headers.get("Content-Length", 0))
                data = self.rfile.read(length)
                part, parts = int(q["part"][0]), int(q["parts"][0])
                if outer.uploads in outer.upload_drops:
                    self.send_response(502)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                outer.part_count = parts
                outer.parts[part] = data
                done = len(outer.parts) == parts
                body = json.dumps({"accepted_into": 0} if done else {"complete": False, "part": part}).encode()
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.sha = "abc123"
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.saved_chunk = group_worker.CHUNK_BYTES
        self.saved_time = group_worker.time
        group_worker.CHUNK_BYTES = 1000
        group_worker.time = types.SimpleNamespace(sleep=lambda s: None, time=self.saved_time.time, monotonic=self.saved_time.monotonic)

    def tearDown(self):
        group_worker.CHUNK_BYTES = self.saved_chunk
        group_worker.time = self.saved_time

    def test_download_reassembles_exactly_after_dropped_chunks(self):
        body = bytes(range(256)) * 20  # 5120 bytes: six chunks
        fake = FakeCoordinator(body, drop_requests={2, 4})
        try:
            got, round_no, sha = group_worker._ranged_download(fake.url, None, "/v1/global?dtype=int8")
        finally:
            fake.close()
        self.assertEqual(got, body)
        self.assertEqual((round_no, sha), (1, "abc123"))
        self.assertGreater(fake.requests, 6, "the dropped chunks should have been asked for again")

    def test_a_global_that_changes_mid_download_is_fetched_again_from_the_new_one(self):
        old, new = b"O" * 3000, b"N" * 3000
        fake = FakeCoordinator(old)
        original_get = fake.server.RequestHandlerClass.do_GET

        def switch_after_first_chunk(handler):
            if fake.requests == 1:
                fake.body, fake.sha = new, "newsha"
            return original_get(handler)

        fake.server.RequestHandlerClass.do_GET = switch_after_first_chunk
        try:
            got, _, sha = group_worker._ranged_download(fake.url, None, "/v1/global?dtype=int8")
        finally:
            fake.close()
        self.assertEqual(got, new, "the first chunk came from the old global and must not be mixed in")
        self.assertEqual(sha, "newsha")

    def test_delta_upload_is_chunked_and_survives_a_failed_chunk(self):
        delta = {"w": torch.arange(1200, dtype=torch.float16).reshape(20, 60), "b": torch.ones(7, dtype=torch.float16)}
        expected = st_save({k: v.contiguous() for k, v in delta.items()})
        fake = FakeCoordinator(b"", )
        fake.upload_drops = {1}  # the first chunk fails with a gateway error and is sent again
        try:
            code, body = group_worker.post_delta(fake.url, None, "g0", 1, 3, delta)
        finally:
            fake.close()
        self.assertEqual(code, 202)
        self.assertEqual(body["accepted_into"], 0)
        self.assertEqual(fake.part_count, -(-len(expected) // 1000))
        reassembled = b"".join(fake.parts[i] for i in sorted(fake.parts))
        self.assertEqual(reassembled, expected)


if __name__ == "__main__":
    unittest.main()
