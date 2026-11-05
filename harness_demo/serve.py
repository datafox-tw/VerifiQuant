"""Start the demo server from any working directory: python3 harness_demo/serve.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness_demo.app import app  # noqa: E402

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=6230, debug=False)
