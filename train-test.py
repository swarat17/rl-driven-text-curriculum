from sklearn.model_selection import train_test_split
import numpy as np

data = np.load("bert_embeddings.npz")
X, y = data['embeddings'], data['labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.savez("bert_embeddings_train.npz", embeddings=X_train, labels=y_train)
np.savez("bert_embeddings_test.npz", embeddings=X_test, labels=y_test)