"""Match a target against what we already know — before scanning it.

Three sources, in this order:

1. **the local product DB** (``nsfw.sqlite``) - free and offline: a domain
   is looked up in ``known_domains``, a local file by its sha256 in
   ``media_hashes``;
2. **a hash list file** - free and offline: a text file with one sha256 per
   line and an optional reason after it (``sha256  reason``);
3. **the hosted API** - metered, so it is opt-in: ``--api-url`` plus
   ``--api-key``.

``--ignore-db`` skips all of it and scans the target anyway.

The API is asked in the site's own vocabulary: ``?p=api&action=lookup&domain=``
for a domain and ``?p=api&action=hash&sha256=`` for bytes.  The hash
endpoint is a paid/higher-tier action; without a key it is not used.
"""

import hashlib
import json
import os
import sqlite3
from urllib.parse import urlparse

__all__ = ["KnownTargets", "sha256_file", "read_hash_list"]


def sha256_file(path, chunk_size=1048576):
    """Hex sha256 of a file, streamed.  None when unreadable."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def read_hash_list(path):
    """Reads a hash list file into ``{sha256: reason}``.

    Format: one sha256 per line, an optional reason after it, ``#``
    comments and blank lines ignored.
    """
    out = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                digest = parts[0].strip().lower()
                if len(digest) != 64:
                    continue
                out[digest] = parts[1].strip() if len(parts) > 1 else None
    except OSError:
        return {}
    return out


def _domain_of(target):
    """The host of a URL (None for a local path)."""
    text = (target or "").strip()
    if not text or os.path.exists(text):
        return None
    if text.lower().startswith("file://"):
        return None
    if not text.lower().startswith(("http://", "https://")):
        text = "https://" + text
    try:
        host = urlparse(text).hostname
    except ValueError:
        return None
    return host.lower().rstrip(".") if host else None


class KnownTargets:
    """Answers "do we already know this target, and how is it rated?"."""

    def __init__(self, db_path=None, hash_list=None, api_url=None,
                 api_key=None, timeout=10):
        self.db_path = db_path
        self.hash_list = hash_list
        self.api_url = (api_url or "").rstrip("/") if api_url else None
        self.api_key = api_key
        self.timeout = timeout
        self._hashes = None
        self._db = None

    # -- helpers ------------------------------------------------------
    @property
    def hashes(self):
        if self._hashes is None:
            self._hashes = read_hash_list(self.hash_list) if self.hash_list else {}
        return self._hashes

    @property
    def db(self):
        if self._db is None:
            if not self.db_path or not os.path.isfile(self.db_path):
                return None
            try:
                # a proper file URI (Windows paths need forward slashes)
                uri = "file:///%s?mode=ro" % (
                    os.path.abspath(self.db_path).replace("\\", "/")
                    .lstrip("/"))
                conn = sqlite3.connect(uri, uri=True, timeout=self.timeout)
                conn.row_factory = sqlite3.Row
                self._db = conn
            except sqlite3.Error:
                return None
        return self._db

    def close(self):
        if self._db is not None:
            try:
                self._db.close()
            except sqlite3.Error:
                pass
            self._db = None

    # -- lookups ------------------------------------------------------
    def lookup_hash(self, digest):
        """Looks a sha256 up: hash list -> local DB -> API."""
        if not digest:
            return None
        digest = digest.lower()
        if digest in self.hashes:
            # the list says "we already know these bytes" - the reason is
            # what the list carries, the rating may be unknown
            return {"known": True, "rating": None, "source": "hash-list",
                    "reason": self.hashes[digest], "blocked": None}
        db = self.db
        if db is not None:
            try:
                # SELECT * keeps this working on an older DB (the reason
                # column may not be there yet)
                row = db.execute(
                    "SELECT * FROM media_hashes WHERE hash = ?",
                    (digest,)).fetchone()
            except sqlite3.Error:
                row = None
            if row is not None:
                keys = row.keys()
                return {"known": True, "rating": row["rating"],
                        "source": "local-db",
                        "reason": row["reason"] if "reason" in keys else None,
                        "media_type": row["media_type"]
                        if "media_type" in keys else None,
                        "blocked": (row["rating"] or 0) >= 10}
        return self._api("hash", {"sha256": digest})

    def lookup_domain(self, domain):
        """Looks a domain up: local DB -> API."""
        if not domain:
            return None
        db = self.db
        if db is not None:
            try:
                row = db.execute(
                    "SELECT * FROM known_domains WHERE domain = ?",
                    (domain,)).fetchone()
            except sqlite3.Error:
                row = None
            if row is not None:
                keys = row.keys()
                return {"known": True, "rating": row["rating"],
                        "source": row["source"] if "source" in keys else "known",
                        "reason": row["reason"] if "reason" in keys else None,
                        "blocked": (row["rating"] or 0) >= 10}
        return self._api("lookup", {"domain": domain})

    def lookup(self, target):
        """Looks a target up (a local file by hash, a URL by domain)."""
        text = (target or "").strip()
        if not text:
            return None
        path = text[7:] if text.lower().startswith("file://") else text
        if os.path.isfile(path):
            return self.lookup_hash(sha256_file(path))
        domain = _domain_of(text)
        if domain:
            return self.lookup_domain(domain)
        return None

    # -- the paid API --------------------------------------------------
    def _api(self, action, params):
        if not self.api_url or not self.api_key:
            return None
        try:
            import requests
        except ImportError:
            return None
        query = "&".join("%s=%s" % (key, value)
                         for key, value in params.items())
        url = "%s/?p=api&action=%s&%s" % (self.api_url, action, query)
        headers = {"Authorization": "Bearer %s" % self.api_key,
                   "X-Api-Key": self.api_key}
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            if response.status_code != 200:
                return None
            data = response.json()
        except Exception:
            return None
        if not isinstance(data, dict) or data.get("rating") is None:
            return None
        return {"known": True, "rating": data.get("rating"),
                "source": "api:%s" % action,
                "reason": data.get("reason"),
                "blocked": bool(data.get("blocked"))}
