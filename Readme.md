# 🧠 Curriculum Learning for Text Classification via PPO-Based Difficulty Estimation

This repository implements a reinforcement learning-guided curriculum learning pipeline for efficient and robust text classification using pretrained language model embeddings (e.g., BERT). The core idea is to estimate sample difficulty using a PPO agent and train classifiers progressively on increasingly harder samples.

---

## 📁 Project Structure

```
├── cache_embeddings.py             # Extract BERT embeddings and cache them to disk
├── train-test.py                   # Train/test split and save for further processing
├── train_estimator_rl.ipynb        # Train PPO agent and regression-based difficulty estimator
├── sort_and_bucket_difficulty.py   # Sort samples by difficulty and split into curriculum buckets
├── test_estimator.ipynb            # Train and evaluate classifiers using different curriculum strategies
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
train_estimator_rl.ipynb
```
- Trains a PPO agent using Gym-style environment
- Converts agent rollouts into difficulty scores
- Trains a regressor on these scores
- Saves `difficulty_scores_rl.npz` and `rl_trained_estimator.pth`

---

### ✅ Step 4: Sort and Bucket Data
```bash
python sort_and_bucket_difficulty.py
```
- Uses difficulty scores to divide data into easy, medium, and hard buckets
- Saves bucketed train sets

---

### ✅ Step 5: Train Classifiers Using Curriculum
Open and run:

```bash
test_estimator.ipynb
```
- Trains classifiers using:
  - Vanilla (full data at once)
  - PPO-based curriculum
  - Random curriculum
  - Blended curriculum
- Plots training metrics and evaluates test accuracy
- Saves visualizations and optionally confusion matrices

---

## 📊 Results Summary

- Curriculum learning using PPO-estimated difficulty scores improves generalization and training efficiency.
- Achieved up to **0.6% higher test accuracy** while **reducing training time by over 50%** compared to vanilla training.
- All experiments are reproducible with saved checkpoints and logs.

---

## 🧠 Citation (if applicable)

> This project is part of academic work on curriculum learning for LLM-based NLP tasks. Please credit appropriately if used.

---

## 🙋 Questions or Contributions?

Feel free to open issues or pull requests if you'd like to contribute or adapt this pipeline to other datasets or models!