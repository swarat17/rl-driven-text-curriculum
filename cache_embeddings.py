# step1_cache_embeddings.py
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

# Load model/tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
model.eval()

# Load dataset
dataset = load_dataset("ag_news", split="train[:50000]")  # limit for RAM

embeddings = []
labels = []
texts = []

with torch.no_grad():
    for example in tqdm(dataset):
        text = example['text']
        tokens = tokenizer(text, truncation=True, padding='max_length', max_length=64, return_tensors='pt').to(device)
        output = model(**tokens).last_hidden_state[:, 0, :]  # CLS token embedding
        embeddings.append(output.squeeze(0).cpu().numpy())
        labels.append(example['label'])
        texts.append(text)

np.savez("bert_embeddings.npz", embeddings=np.array(embeddings), labels=np.array(labels), texts=np.array(texts))