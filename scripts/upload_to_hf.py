import os
from huggingface_hub import HfApi, Repository

def upload_model():
    api = HfApi()
    repo_id = "timlawrenz/neural-universal-machine-dit"
    
    print(f"Creating repository {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
    except Exception as e:
        print(f"Repo exists or error: {e}")
        
    print("Uploading Epoch 340 pre-trained checkpoint...")
    api.upload_file(
        path_or_fileobj="checkpoints/num_dit_epoch_340.pt",
        path_in_repo="num_dit_epoch_340.pt",
        repo_id=repo_id,
        commit_message="Add 128-node Continuous Pre-Trained DiT Checkpoint (Epoch 340)"
    )
    print("Upload complete!")

if __name__ == "__main__":
    upload_model()
