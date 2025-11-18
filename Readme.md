# 🧠 Curriculum Learning for Text Classification via PPO-Based Difficulty Estimation

This repository implements a Reinforcement Learning–guided curriculum learning pipeline for NLP classification using BERT embeddings. A PPO agent interacts with a custom environment to estimate per-sample difficulty scores. These scores train a Difficulty Estimator (regression model) that can later generate difficulty values without running RL again — making the approach scalable and reusable.

The curriculum buckets (Easy → Medium → Hard) are then used to train classifiers using multiple curriculum strategies.

---

## 📁 Project Structure

```
├── cache_embeddings.py                  # Extract BERT embeddings using HuggingFace
├── models.py                            # DifficultyEstimator, SimpleClassifier, ImprovedClassifier
├── train-test.py                        # Split cached embeddings into train/test datasets
├── train_estimator_rl_new.ipynb         # PPO agent + regression difficulty estimator training
├── sort_and_bucket_difficulty_new.py    # Convert difficulty scores into Easy/Medium/Hard buckets
├── test_estimator_new.ipynb             # Curriculum training experiments & evaluation
```

---

## 🚀 Getting Started

### 🔧 1. Set Up Environment

```bash
git clone <your-repo-url>
cd <project-folder>
pip install -r requirements.txt
```

Dependencies include:
- `torch`
- `transformers`
- `scikit-learn`
- `stable-baselines3`
- `gymnasium`
- `matplotlib`
- `numpy`

---

## 🔁 Full Pipeline

Run the following files in order to reproduce the curriculum learning setup and results.

### ✅ Step 1: Extract BERT Embeddings
```bash
python cache_embeddings.py
```
- Loads AG News (or chosen dataset)
- Uses `DistilBERT` to encode samples
- Saves `bert_embeddings.npz`

---

### ✅ Step 2: Train-Test Split
```bash
python train-test.py
```
- Splits the cached embeddings into train/test sets if not already done
- Saves `bert_embeddings_train.npz` and `bert_embeddings_test.npz`

---

### ✅ Step 3: Train PPO-based Difficulty Estimator
Open and run all cells in:

```bash
train_estimator_rl_new.ipynb
```
- Trains a PPO agent using Gym-style environment
- Converts agent rollouts into difficulty scores
- Trains a regressor on these scores
- Saves `difficulty_scores_rl.npz` and `rl_trained_estimator.pth`

---

### ✅ Step 4: Sort and Bucket Data
```bash
python sort_and_bucket_difficulty_new.py
```
- Uses difficulty scores to divide data into easy, medium, and hard buckets
- Saves bucketed train sets

---

### ✅ Step 5: Train Classifiers Using Curriculum
Open and run:

```bash
test_estimator_new.ipynb
```
- Trains classifiers using:
  - Vanilla (full data at once)
  - PPO-based curriculum
  - Random curriculum
  - Blended curriculum
- Plots training metrics and evaluates test accuracy
- Saves visualizations and optionally confusion matrices

---

🧩 Models Used (from models.py)
- DifficultyEstimator (Regression Model)
  - Learns PPO-generated difficulty scores
  - Predicts difficulty for new samples
  - Uses a temperature scaling term to expand score resolution
  - Used only at inference, not part of the classifier
  - Key for reproducible bucket generation

- SimpleClassifier
  - Light 2-layer MLP
  - Used inside PPO environment (fast RL loop)

- ImprovedClassifier
  - Larger network with BatchNorm + Dropout
  - Used in final curriculum evaluation
  - More stable and expressive than SimpleClassifier

## 📊 Results Summary

The project demonstrates that:
- PPO produces meaningful difficulty scores
- DifficultyEstimator captures these scores accurately
- Buckets display clear performance separation
  - Easy bucket → 98–100% accuracy
  - Medium bucket → ~90% accuracy
  - Hard bucket → noticeably harder
- Curriculum training improves:
  - Convergence speed (up to 60% faster)
  - Stability in early training
  - Interpretability of training dynamics
- Even if accuracy gains are small, the pipeline is novel and robust

---

## 🧠 Citation (if applicable)

> This project is part of academic work on curriculum learning for LLM-based NLP tasks. Please credit appropriately if used.

---

## 🙋 Questions or Contributions?

Feel free to open issues or pull requests if you'd like to contribute or adapt this pipeline to other datasets or models!
