"""Polite, resumable downloader with a provenance manifest.

Every file is written once to `03_data/raw/<source>/...`; the manifest `03_data/raw/<source>/MANIFEST.csv` records
url, relative path, HTTP status, bytes, SHA-256 and UTC retrieval time. A file already listed with status 200 and a
matching size on disk is not downloaded again. Requests are sequential with a delay (public government servers).

TLS: some Indian government servers send an incomplete certificate chain that Python cannot verify. Such hosts are
listed in `INCOMPLETE_CHAIN_HOSTS`; for them verification is disabled and every download is still fingerprinted with
SHA-256 so that the retrieved bytes are auditable.
"""
import csv
import datetime as dt
import hashlib
import json
import os
import threading
import time
from urllib.parse import urlparse

import requests
import urllib3

from .paths import RAW

INCOMPLETE_CHAIN_HOSTS = {"webapi.grid-india.in", "webcdn.grid-india.in", "grid-india.in", "npp.gov.in"}
USER_AGENT = "Mozilla/5.0 (academic research; public government data; public research)"
FIELDS = ["url", "path", "status", "bytes", "sha256", "retrieved_utc", "note"]
_lock = threading.Lock()
urllib3.disable_warnings()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Fetcher:
    def __init__(self, source, delay=0.6, timeout=120, retries=4):
        self.source = source
        self.dir = os.path.join(RAW, source)
        os.makedirs(self.dir, exist_ok=True)
        self.manifest_path = os.path.join(self.dir, "MANIFEST.csv")
        self.delay, self.timeout, self.retries = delay, timeout, retries
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})
        self.done = {}
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    self.done[r["url"]] = r
        self._last = 0.0

    def _verify(self, url):
        return urlparse(url).hostname not in INCOMPLETE_CHAIN_HOSTS

    def _wait(self):
        gap = time.time() - self._last
        if gap < self.delay:
            time.sleep(self.delay - gap)
        self._last = time.time()

    def _record(self, row):
        with _lock:
            new = not os.path.exists(self.manifest_path)
            with open(self.manifest_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            self.done[row["url"]] = row

    def have(self, url):
        r = self.done.get(url)
        if not r or r["status"] != "200":
            return False
        p = os.path.join(self.dir, r["path"])
        return os.path.exists(p) and os.path.getsize(p) == int(r["bytes"])

    def get(self, url, relpath, note="", min_bytes=1):
        """Download url to <source>/<relpath>. Returns absolute path or None (status recorded)."""
        if self.have(url):
            return os.path.join(self.dir, self.done[url]["path"])
        if url in self.done and self.done[url]["status"] == "404":
            return None
        dst = os.path.join(self.dir, relpath)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        status, err = None, ""
        for attempt in range(self.retries):
            self._wait()
            try:
                r = self.s.get(url, timeout=self.timeout, verify=self._verify(url))
                status = r.status_code
                if status == 200 and b"portal_url" in r.content[:2000] and b"<HTML>" in r.content[:20].upper():
                    # local network login page intercepting the request (seen on 14 Sep 2026): never store it
                    status, err = "captive_portal", "network login page"
                    time.sleep(60)
                    continue
                if status == 200 and len(r.content) >= min_bytes:
                    tmp = dst + ".part"
                    with open(tmp, "wb") as fh:
                        fh.write(r.content)
                    os.replace(tmp, dst)
                    self._record(dict(url=url, path=relpath.replace("\\", "/"), status="200", bytes=len(r.content),
                                      sha256=hashlib.sha256(r.content).hexdigest(),
                                      retrieved_utc=dt.datetime.utcnow().isoformat(timespec="seconds"), note=note))
                    return dst
                if status in (404, 410):
                    break
                if status == 200:
                    status, err = "empty", f"{len(r.content)} bytes"
                    break
            except requests.RequestException as e:
                err = type(e).__name__
                time.sleep(5 * (attempt + 1))
        self._record(dict(url=url, path=relpath.replace("\\", "/"), status=str(status or "error"), bytes=0, sha256="",
                          retrieved_utc=dt.datetime.utcnow().isoformat(timespec="seconds"), note=(note + " " + err).strip()))
        return None

    def post_json(self, url, payload):
        for attempt in range(self.retries):
            self._wait()
            try:
                r = self.s.post(url, data=json.dumps(payload), timeout=self.timeout, verify=self._verify(url),
                                headers={"Content-Type": "application/json", "Origin": "https://grid-india.in",
                                         "Referer": "https://grid-india.in/"})
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"POST failed after retries: {url} {payload}")
