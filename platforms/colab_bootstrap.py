"""Paste into a Colab cell (GPU runtime). Clones, installs and runs the pipeline."""
import os
import subprocess

REPO = os.environ.get("REPO_URL", "https://github.com/<org>/support-ticket-triage.git")  # set me

try:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
except ImportError:
    print("Not on Colab; skipping Drive mount.")

if not os.path.isdir("support-ticket-triage"):
    subprocess.run(["git", "clone", REPO], check=True)
os.chdir("support-ticket-triage")
subprocess.run(["pip", "install", "-q", "-e", ".[train]"], check=True)
for step in ("data", "train", "predict", "evaluate"):
    subprocess.run(
        ["python", "-m", "triage.cli", step, "--config", "configs/platform/colab.yaml"],
        check=True,
    )
