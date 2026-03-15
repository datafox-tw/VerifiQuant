from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifiquant_v2.preprocessing.dataset_case_to_fic import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
