from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifiquant.pipeline.run_framework_guided_self_improve_pipeline import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
