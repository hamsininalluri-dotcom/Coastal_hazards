"""CLI: python -m src.priority.cli  -> prints the explained priority for every cluster in the test split."""
import subprocess
import sys

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "src.pipeline", "--split", "test", "--top", "12"]))
