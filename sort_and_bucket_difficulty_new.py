import torch
import numpy as np
from models import DifficultyEstimator  # Ensure this is correctly imported

def score_and_bucket(embedding_path, estimator_path, output_path, device='cuda'):
    print("Loading data...")
    data = np.load(embedding_path)
    embeddings = torch.tensor(data["embeddings"], dtype=torch.float32).to(device)
    labels = data["labels"]

    print("Loading difficulty estimator...")
    model = DifficultyEstimator()
    model.load_state_dict(torch.load(estimator_path, map_location=device))
    model.to(device)
    model.eval()

    print("Scoring...")
    with torch.no_grad():
        scores = model(embeddings).squeeze().cpu().numpy()

    # Normalize scores
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    print("Bucketing...")
    n = len(scores)
    sorted_idx = np.argsort(scores)
    easy_idx = sorted_idx[:n // 3]
    medium_idx = sorted_idx[n // 3:2 * n // 3]
    hard_idx = sorted_idx[2 * n // 3:]

    np.savez(output_path,
             easy_embeddings=embeddings[easy_idx].cpu().numpy(),
             easy_labels=labels[easy_idx],
             medium_embeddings=embeddings[medium_idx].cpu().numpy(),
             medium_labels=labels[medium_idx],
             hard_embeddings=embeddings[hard_idx].cpu().numpy(),
             hard_labels=labels[hard_idx])
    print(f"Saved difficulty buckets to {output_path}")


# Example usage
if __name__ == "__main__":
    score_and_bucket(
        embedding_path="bert_embeddings_train.npz",
        estimator_path="rl_trained_estimator_new.pth",
        output_path="rl_buckets.npz"
    )