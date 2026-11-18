import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from nsfw_scanner.scanner import main as scanner_main
from nsfw_scanner.webservice import WebserviceError, submit_media_file


class _SampleHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        self.requests.append((self.path, self.headers["Content-Type"], body))
        response = json.dumps({
            "hash": "a" * 64,
            "rating": 7,
            "verdict": "revealing",
            "stored": True,
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args):
        pass


class WebserviceSubmissionTests(unittest.TestCase):
    def setUp(self):
        _SampleHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _SampleHandler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = "http://127.0.0.1:%s" % self.server.server_port
        self.tempdir = tempfile.TemporaryDirectory()
        self.media_path = os.path.join(self.tempdir.name, "sample.mp4")
        with open(self.media_path, "wb") as media_file:
            media_file.write(b"small media fixture")

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.tempdir.cleanup()

    def test_submit_media_file_posts_multipart_and_reads_hosted_rating(self):
        result = submit_media_file(self.url, self.media_path)

        self.assertEqual(result["rating"], 7)
        request_path, content_type, body = _SampleHandler.requests[0]
        self.assertEqual(request_path, "/v1/sample")
        self.assertIn("multipart/form-data; boundary=", content_type)
        self.assertIn(b'name="file"; filename="sample.mp4"', body)
        self.assertIn(b"small media fixture", body)

    def test_submit_media_file_includes_valid_imdb_id(self):
        submit_media_file(self.url, self.media_path, imdb_id="tt0111161")
        _path, _content_type, body = _SampleHandler.requests[0]
        self.assertIn(b'name="imdb_id"', body)
        self.assertIn(b"tt0111161", body)

    def test_submit_media_file_drops_invalid_imdb_id(self):
        submit_media_file(self.url, self.media_path, imdb_id="not-an-id")
        _path, _content_type, body = _SampleHandler.requests[0]
        self.assertNotIn(b'name="imdb_id"', body)

    def test_cli_option_keeps_local_rating_and_adds_hosted_result(self):
        output_path = os.path.join(self.tempdir.name, "result.json")
        with patch("nsfw_scanner.scanner.set_model_mode"), patch(
                "nsfw_scanner.scanner.rate", return_value={"rating": 6}):
            scanner_main([
                self.media_path,
                "--ignore-db",
                "--submit-to-webservice", self.url,
                "--output", output_path,
            ])

        with open(output_path, encoding="utf-8") as result_file:
            result = json.load(result_file)
        self.assertEqual(result["rating"], 6)
        self.assertEqual(result["webservice_submission"]["rating"], 7)
        self.assertEqual(_SampleHandler.requests[0][0], "/v1/sample")

    def test_non_local_http_service_is_rejected(self):
        with self.assertRaises(WebserviceError):
            submit_media_file("http://example.com", self.media_path)

    def test_cli_rejects_submission_with_benchmark_modes(self):
        with patch("nsfw_scanner.scanner.set_model_mode"), patch(
                "nsfw_scanner.scanner.rate", return_value={"rating": 6}):
            with self.assertRaises(SystemExit) as error:
                scanner_main([
                    self.media_path,
                    "--submit-to-webservice", self.url,
                    "--benchmark",
                ])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
