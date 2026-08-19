"""The `/metrics` endpoint a Prometheus server scrapes.

Deliberately the standard library's `ThreadingHTTPServer` rather than adding a
web framework: this serves two paths and must not become a place where features
accumulate. It runs on a daemon thread so a worker shutting down is not held
open by it.

Two endpoints, and the second matters more than it looks:

* `/metrics` — the Prometheus exposition format.
* `/healthz` — a liveness answer that does *not* depend on Temporal being
  reachable. A worker that cannot reach Temporal is broken, but restarting it
  will not help; a liveness probe that fails on a Temporal outage turns one
  outage into a crash-loop across every replica at once.

Anything else is a 404. The server binds only the port and holds no state, so
there is nothing here for an attacker to reach — but the executing worker still
gets no ingress rule beyond the scrape, and the NetworkPolicy in `k8s/` says so.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

from etl_migrator.observability.logging import get_logger

log = get_logger(__name__)

METRICS_PATH = "/metrics"
HEALTH_PATH = "/healthz"


def _handler_for(registry: CollectorRegistry) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        # The default logs every request to stderr, bypassing structlog and
        # producing a line per scrape — one every fifteen seconds, for ever.
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == METRICS_PATH:
                self._respond(200, generate_latest(registry), CONTENT_TYPE_LATEST)
            elif path == HEALTH_PATH:
                self._respond(200, b"ok\n", "text/plain; charset=utf-8")
            else:
                self._respond(404, b"not found\n", "text/plain; charset=utf-8")

    return Handler


class MetricsServer:
    """A scrape endpoint on a daemon thread."""

    def __init__(self, registry: CollectorRegistry, *, port: int, host: str = "0.0.0.0"):
        # Binding all interfaces is correct in a container: the pod's address is
        # not known ahead of time and the NetworkPolicy is what limits who can
        # reach it. Binding localhost would make the pod unscrapeable.
        self._server = ThreadingHTTPServer((host, port), _handler_for(registry))
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """The bound port. Differs from the requested one when 0 was asked for,
        which is what tests use to avoid fighting over a fixed port."""
        return int(self._server.server_address[1])

    def start(self) -> MetricsServer:
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="metrics", daemon=True
        )
        self._thread.start()
        log.info("metrics.serving", port=self.port, path=METRICS_PATH)
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> MetricsServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
