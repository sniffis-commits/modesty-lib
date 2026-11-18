"""
Frame Bus — HTTP endpoint exposing frames from the ring buffer during a scan.

The frame bus lives only for the duration of a single scan (transient, no disk
storage). It allows the website or a debugging UI to show "problem frames"
without re-decoding the video.

Usage:
    from nsfw_scanner.frame_bus import FrameBus
    
    bus = FrameBus(ring_name="nsfw_scan_12345")
    bus.start(port=0)  # picks random free port
    print(f"Frame bus at http://localhost:{bus.port}/frame/42")
    
    # During scan, frames are available at /frame/<seq>
    # After scan, bus.stop() shuts down the HTTP server

Architecture:
    ┌─────────────┐     ring buffer      ┌─────────────┐     HTTP      ┌──────────┐
    │  ffmpeg     │ ──────────────────►  │  frame_bus  │ ───────────►  │  browser │
    │  (writer)   │   (shared memory)    │  (reader)   │   /frame/N    │  /site   │
    └─────────────┘                      └─────────────┘               └──────────┘
"""

import http.server
import json
import socket
import threading
import time
from http import HTTPStatus
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs

try:
    from .buffered_transport import BufferedTransport, Frame as TransportFrame
    _BUFFERED_AVAILABLE = True
except ImportError:
    _BUFFERED_AVAILABLE = False

class FrameBus:
    """HTTP server exposing frames from a scan's ring buffer."""
    
    def __init__(self, ring_name: str = "nsfw_frame_bus", host: str = "127.0.0.1", port: int = 0):
        self.ring_name = ring_name
        self.host = host
        self.port = port
        self._server: Optional[http.server.HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._frames: Dict[int, TransportFrame] = {}
        self._lock = threading.Lock()
        self._running = False
        self._transport: Optional[BufferedTransport] = None
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    def start(self) -> int:
        """Start the HTTP server. Returns the actual port."""
        if self._running:
            return self.port
        
        # Find free port if port=0
        if self.port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.host, 0))
                self.port = s.getsockname()[1]
        
        self._server = http.server.HTTPServer((self.host, self.port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._running = True
        return self.port
    
    def stop(self):
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._running = False
        self._frames.clear()
    
    def register_frame(self, frame: TransportFrame):
        """Called by the scanner to register a frame."""
        with self._lock:
            self._frames[frame.seq] = frame
    
    def get_frame(self, seq: int) -> Optional[TransportFrame]:
        """Get a frame by sequence number."""
        with self._lock:
            return self._frames.get(seq)
    
    def list_frames(self, limit: int = 100) -> list:
        """List recent frames."""
        with self._lock:
            items = sorted(self._frames.items(), key=lambda x: x[0], reverse=True)
            return [{
                "seq": seq,
                "pts_us": f.pts_us,
                "width": f.width,
                "height": f.height,
                "format": f.format,
                "stride": f.stride,
                "payload_size": len(f.payload),
            } for seq, f in items[:limit]]
    
    def _make_handler(self):
        bus = self
        
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                
                if path == "/health":
                    self._send_json({"status": "ok", "ring": bus.ring_name, "frames": len(bus._frames)})
                elif path == "/frames":
                    query = parse_qs(parsed.query)
                    limit = int(query.get("limit", ["100"])[0])
                    self._send_json({"frames": bus.list_frames(limit)})
                elif path.startswith("/frame/"):
                    try:
                        seq = int(path.split("/")[-1])
                        frame = bus.get_frame(seq)
                        if frame:
                            # Return frame metadata + base64 payload (for small frames)
                            import base64
                            payload_b64 = base64.b64encode(frame.payload).decode() if len(frame.payload) < 50000 else None
                            self._send_json({
                                "seq": frame.seq,
                                "pts_us": frame.pts_us,
                                "width": frame.width,
                                "height": frame.height,
                                "format": frame.format,
                                "stride": frame.stride,
                                "payload_b64": payload_b64,
                                "payload_size": len(frame.payload),
                            })
                        else:
                            self._send_error(HTTPStatus.NOT_FOUND, f"Frame {seq} not found")
                    except ValueError:
                        self._send_error(HTTPStatus.BAD_REQUEST, "Invalid sequence number")
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            
            def _send_json(self, data: Dict[str, Any]):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            
            def _send_error(self, status: HTTPStatus, message: str):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": message}).encode())
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        return Handler


class FrameBusManager:
    """Manages multiple frame buses for concurrent scans."""
    
    def __init__(self):
        self._buses: Dict[str, FrameBus] = {}
        self._lock = threading.Lock()
    
    def create_bus(self, scan_id: str) -> FrameBus:
        """Create a new frame bus for a scan."""
        with self._lock:
            if scan_id in self._buses:
                self._buses[scan_id].stop()
            bus = FrameBus(ring_name=f"nsfw_scan_{scan_id}")
            bus.start()
            self._buses[scan_id] = bus
            return bus
    
    def get_bus(self, scan_id: str) -> Optional[FrameBus]:
        with self._lock:
            return self._buses.get(scan_id)
    
    def remove_bus(self, scan_id: str):
        with self._lock:
            if scan_id in self._buses:
                self._buses[scan_id].stop()
                del self._buses[scan_id]
    
    def shutdown_all(self):
        with self._lock:
            for bus in self._buses.values():
                bus.stop()
            self._buses.clear()


# Global manager instance
_manager: Optional[FrameBusManager] = None

def get_frame_bus_manager() -> FrameBusManager:
    global _manager
    if _manager is None:
        _manager = FrameBusManager()
    return _manager