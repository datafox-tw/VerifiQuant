from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifiquant.preprocessing.finchain_def_to_jsoncard import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
