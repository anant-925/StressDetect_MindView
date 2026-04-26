"""Download the pretrained checkpoint from HuggingFace Hub."""
from huggingface_hub import hf_hub_download
import shutil, os

path = hf_hub_download(
    repo_id="Ace-119/stress-detection-cnn",
    filename="model.pt",
)

os.makedirs("checkpoints", exist_ok=True)
shutil.copy(path, "checkpoints/model.pt")

print("Model saved to checkpoints/model.pt")
