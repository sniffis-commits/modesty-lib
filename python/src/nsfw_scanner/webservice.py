"""Opt-in media submissions to the Modesty webservice."""

import http.client
import json
import mimetypes
import os
import re
import secrets
from urllib.parse import urlsplit


class WebserviceError(RuntimeError):
    """A media submission could not be completed."""


def submit_media_file(service_url, file_path, timeout=600, imdb_id=None):
    """Upload a local media file to ``POST /v1/sample`` for hosted scanning.

    The webservice scans the uploaded bytes itself; this does not submit the
    caller's locally computed rating. Uploads are opt-in because media leaves
    the device.

    ``imdb_id`` (optional, e.g. ``tt0111161``) tags the submission with a
    catalog title so the result can be looked up via ``GET /v1/title/...``.
    Invalid values are dropped rather than sent.
    """
    parsed = urlsplit((service_url or "").strip())
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or
            parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise WebserviceError(
            "Webservice URL must be an HTTP(S) base URL without credentials, "
            "query, or fragment."
        )
    if parsed.scheme != "https" and parsed.hostname not in (
            "localhost", "127.0.0.1", "::1"):
        raise WebserviceError("Use HTTPS for non-local webservice uploads.")

    path = os.path.abspath(file_path)
    if not os.path.isfile(path):
        raise WebserviceError("Only an existing local media file can be submitted.")
    filename = "".join(
        character for character in os.path.basename(path)
        if character not in '"\r\n'
    ) or "media-upload"
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    boundary = "----ModestyBoundary" + secrets.token_hex(16)
    imdb_field = b""
    if imdb_id and re.match(r"^tt\d+$", str(imdb_id).strip()):
        imdb_field = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="imdb_id"\r\n\r\n'
            f"{str(imdb_id).strip()}\r\n"
        ).encode("utf-8")
    prefix = imdb_field + (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    content_length = len(prefix) + os.path.getsize(path) + len(suffix)
    request_path = (parsed.path.rstrip("/") + "/v1/sample") or "/v1/sample"
    connection_type = (http.client.HTTPSConnection
                       if parsed.scheme == "https"
                       else http.client.HTTPConnection)
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    try:
        connection.putrequest("POST", request_path)
        connection.putheader("Content-Type",
                             f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(content_length))
        connection.putheader("Accept", "application/json")
        connection.endheaders()
        connection.send(prefix)
        with open(path, "rb") as media_file:
            while True:
                chunk = media_file.read(1024 * 1024)
                if not chunk:
                    break
                connection.send(chunk)
        connection.send(suffix)
        response = connection.getresponse()
        body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise WebserviceError("Webservice response exceeded 1 MB.")
        try:
            result = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WebserviceError(
                f"Webservice returned invalid JSON (HTTP {response.status})."
            ) from error
        if response.status < 200 or response.status >= 300:
            message = result.get("error") if isinstance(result, dict) else None
            raise WebserviceError(
                message or f"Webservice returned HTTP {response.status}."
            )
        if not isinstance(result, dict):
            raise WebserviceError("Webservice returned an invalid result.")
        rating = result.get("rating")
        if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 10:
            message = result.get("error")
            raise WebserviceError(
                message or "Webservice could not produce a rating for this file."
            )
        return result
    except (OSError, http.client.HTTPException) as error:
        raise WebserviceError(f"Could not submit media to the webservice: {error}") from error
    finally:
        connection.close()
