import torch
import numpy as np
from models import DifficultyEstimator

def sort_and_bucket(
    embedding_path: str,
    estimator_path: str,
    output_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Sorts and buckets samples based on difficulty scores predicted by a trained estimator.

    Args:
        embedding_path (str): Path to .npz file containing 'embeddings' and 'labels'.
        estimator_path (str): Path to the trained estimator .pth file.
        output_path (str): Path to save the new .npz file with easy/medium/hard buckets.
        device (str): 'cuda' or 'cpu'. Defaults to automatic detection.
    """
    print(f"Loading embeddings from {embedding_path}...")
    data = np.load(embedding_path)
    embeddings = torch.tensor(data["embeddings"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.long)

    print(f"Loading estimator from {estimator_path}...")
    estimator = DifficultyEstimator()
    estimator.load_state_dict(torch.load(estimator_path, map_location=device))
    estimator.to(device)
    estimator.eval()

    print("Scoring difficulties...")
    with torch.no_grad():
        scores = estimator(embeddings.to(device)).squeeze().cpu().numpy()

    print("Sorting and bucketing...")
    sorted_indices = np.argsort(scores)
    n = len(scores)
    easy_idx = sorted_indices[:n // 3]
    medium_idx = sorted_indices[n // 3:2 * n // 3]
    hard_idx = sorted_indices[2 * n // 3:]

    np.savez(output_path,
             easy_embeddings=embeddings[easy_idx].numpy(),
             easy_labels=labels[easy_idx].numpy(),
             medium_embeddings=embeddings[medium_idx].numpy(),
             medium_labels=labels[medium_idx].numpy(),
             hard_embeddings=embeddings[hard_idx].numpy(),
             hard_labels=labels[hard_idx].numpy())
    print(f"Buckets saved to {output_path}")


# Example usage:
if __name__ == "__main__":
    sort_and_bucket(
        embedding_path="bert_embeddings_train.npz",
        estimator_path="rl_trained_estimator.pth",
        output_path="difficulty_buckets.npz"
    )