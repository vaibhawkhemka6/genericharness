"""`python run_frontend.py` - serves `web/` (plain HTML/CSS/JS, no
build step) over local HTTP, the browser-facing half of this project's
local chat UI. The other half is `run_os.py`, which has to be running
separately (default `http://127.0.0.1:7777`) for the page to have anything
to talk to - this script only serves static files, it never imports or
touches `wind/*` itself.

Not served as a `file://` URL on purpose: browsers treat `file://` as an
opaque/`null` origin for CORS purposes, which `wind/os/app.py`'s
`CORSMiddleware` (driven by `OSSettings.cors_origins`, itself read from the
`FRONTEND_ORIGIN` env var) can't allow-list cleanly. Serving from a real
`http://localhost:<port>` origin instead is what `FRONTEND_ORIGIN` in
`.env` already lists (`http://localhost:3000` among them) - so port 3000
below matches that existing config with no `.env` change needed.

Plain `http.server` (stdlib), not another FastAPI app - this is a static
file server with zero business logic, not another OS-layer process.

Run: python run_frontend.py
Then, with `run_os.py` running in another terminal:
  open http://localhost:3000
"""

from __future__ import annotations

import functools
import http.server
import os

PORT = 3000
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def main() -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DIRECTORY)
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"Serving {DIRECTORY} at http://127.0.0.1:{PORT}")
        print("Make sure run_os.py is running too (default http://127.0.0.1:7777).")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping.")


if __name__ == "__main__":
    main()
