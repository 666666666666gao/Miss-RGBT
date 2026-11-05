"""Run the bounded CPU factorial then its independent receipt analysis."""
from pathlib import Path
import subprocess
import sys


ROOT = Path("/home/gaob/Miss-RGBT")


if __name__ == "__main__":
    for name in ["probe_fixed_crop_motion28.py", "analyze_fixed_crop_motion28.py"]:
        subprocess.check_call([sys.executable, "-u", str(ROOT / name)])
