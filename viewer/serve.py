"""HTTP server for the detection viewer that supports UNC network paths.

This server proxies requests to `data_link/*` to the actual data directory,
allowing the viewer to work with network paths on Windows where junctions
don't support UNC paths.

Usage:
    python viewer/serve.py [--port PORT]

The server reads `data_index.json` to find the data_root path.
"""

import argparse
import json
import mimetypes
import os
import sys
import urllib.parse
from functools import partial
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent
INDEX_FILE = VIEWER_DIR / "data_index.json"


def _load_data_root() -> tuple[Path | None, bool]:
    """Load data_root from data_index.json. Returns (path, use_proxy)."""
    if not INDEX_FILE.exists():
        return None, False
    try:
        with open(INDEX_FILE, encoding="utf-8") as f:
            data = json.load(f)
        root = data.get("data_root")
        use_proxy = data.get("use_server_proxy", False)
        return Path(root) if root else None, use_proxy
    except Exception:
        return None, False


class ViewerHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler that proxies data_link/* requests to the actual data directory."""

    def __init__(self, *args, data_root: Path | None = None, **kwargs):
        self.data_root = data_root
        super().__init__(*args, **kwargs)

    def translate_path(self, path: str) -> str:
        """Map URL path to filesystem path, proxying data_link/ to data_root."""
        # Decode URL-encoded path
        path = urllib.parse.unquote(path)

        # Remove leading slash and normalize
        path = path.lstrip("/")

        # Check if this is a data_link request
        if path.startswith("data_link/") and self.data_root:
            # Strip "data_link/" prefix and resolve against data_root
            relative = path[len("data_link/") :]
            return str(self.data_root / relative)

        # Otherwise serve from viewer directory
        return str(VIEWER_DIR / path)

    def do_GET(self):
        """Handle GET requests with proper error handling for network paths."""
        try:
            super().do_GET()
        except (OSError, PermissionError) as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Error accessing file: {e}")

    def send_head(self):
        """Send headers, with special handling for video range requests."""
        path = self.translate_path(self.path)
        f = None

        if os.path.isdir(path):
            # Directory: try index.html or list
            parts = urllib.parse.urlsplit(self.path)
            if not parts.path.endswith("/"):
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                new_parts = (parts[0], parts[1], parts[2] + "/", parts[3], parts[4])
                new_url = urllib.parse.urlunsplit(new_parts)
                self.send_header("Location", new_url)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            for index in ("index.html", "detection_viewer.html"):
                index_path = os.path.join(path, index)
                if os.path.isfile(index_path):
                    path = index_path
                    break
            else:
                return self.list_directory(path)

        ctype = self.guess_type(path)

        # Always open in binary mode for HTTP responses
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            fs = os.fstat(f.fileno())
            content_length = fs.st_size

            # Handle Range requests for video seeking
            range_header = self.headers.get("Range")
            if range_header and ctype.startswith(("video/", "audio/")):
                return self._send_partial_content(f, path, fs, range_header, ctype)

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", ctype)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            # Allow video seeking
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return f
        except Exception:
            f.close()
            raise

    def _send_partial_content(self, f, path, fs, range_header, ctype):
        """Handle HTTP Range requests for video/audio seeking."""
        try:
            # Parse Range header: "bytes=START-END" or "bytes=START-"
            if not range_header.startswith("bytes="):
                f.close()
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return None

            range_spec = range_header[6:]  # Remove "bytes="
            if "-" not in range_spec:
                f.close()
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return None

            start_str, end_str = range_spec.split("-", 1)
            file_size = fs.st_size

            if start_str:
                start = int(start_str)
                end = int(end_str) if end_str else file_size - 1
            else:
                # Suffix range: "-500" means last 500 bytes
                suffix_len = int(end_str)
                start = max(0, file_size - suffix_len)
                end = file_size - 1

            # Clamp to file size
            end = min(end, file_size - 1)
            content_length = end - start + 1

            if start > end or start >= file_size:
                f.close()
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return None

            # Seek to start position
            f.seek(start)

            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-type", ctype)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()

            # Return a wrapper that limits read to content_length
            return _RangeFile(f, content_length)

        except (ValueError, OSError):
            f.close()
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None

    def copyfile(self, source, outputfile):
        """Copy file to output, handling RangeFile wrapper."""
        if isinstance(source, _RangeFile):
            # Limited read for range requests
            remaining = source.remaining
            while remaining > 0:
                chunk_size = min(64 * 1024, remaining)
                data = source.file.read(chunk_size)
                if not data:
                    break
                outputfile.write(data)
                remaining -= len(data)
            source.file.close()
        else:
            super().copyfile(source, outputfile)

    def log_message(self, format, *args):
        """Log with color coding for status."""
        status = args[1] if len(args) > 1 else ""
        if status.startswith("2"):
            color = "\033[92m"  # Green
        elif status.startswith("3"):
            color = "\033[93m"  # Yellow
        elif status.startswith("4") or status.startswith("5"):
            color = "\033[91m"  # Red
        else:
            color = ""
        reset = "\033[0m" if color else ""
        sys.stderr.write(f"{color}{self.address_string()} - {format % args}{reset}\n")


class _RangeFile:
    """Wrapper to limit bytes read from a file for Range requests."""

    def __init__(self, file, remaining: int):
        self.file = file
        self.remaining = remaining

    def read(self, size=-1):
        if size < 0:
            size = self.remaining
        size = min(size, self.remaining)
        data = self.file.read(size)
        self.remaining -= len(data)
        return data

    def close(self):
        self.file.close()


def main():
    parser = argparse.ArgumentParser(description="Serve the detection viewer")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--bind", "-b", default="127.0.0.1", help="Address to bind to")
    args = parser.parse_args()

    # Load data root from index
    data_root, use_proxy = _load_data_root()

    if data_root:
        if use_proxy:
            print(f"Proxying data_link/ → {data_root}")
        else:
            print(f"Data root: {data_root}")
            print("(Using local junction - proxy not required but will work)")
    else:
        print("Warning: No data_index.json found. Run 'python viewer/index_data.py' first.")

    # Create handler with data_root
    handler = partial(ViewerHTTPHandler, data_root=data_root, directory=str(VIEWER_DIR))

    # Start server
    os.chdir(VIEWER_DIR)
    with HTTPServer((args.bind, args.port), handler) as httpd:
        print(f"\nViewer: http://{args.bind}:{args.port}/detection_viewer.html")
        print("Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
