import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

def visualize_embeddings(conversations: list, method: str = "tsne"):
    embeddings = [pickle.loads(conv.embedding) for conv in conversations if conv.embedding]

    if not embeddings:
        raise ValueError("Embeddingが空です")

    try:
        embeddings = np.vstack(embeddings)
    except Exception as e:
        raise ValueError(f"embeddingの変換に失敗しました: {e}")

    if method == "tsne":
        n_samples = len(embeddings)
        if n_samples < 2:
            raise ValueError("t-SNEを実行するには最低2件のembeddingが必要です")
        # perplexity must be < n_samples
        perplexity = min(30, n_samples - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("methodは'tsne'または'pca'を指定してください")

    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c='blue', alpha=0.5)
    plt.title(f"{method.upper()} Clustering Visualization")

    if method == "tsne":
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
    else:
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")


    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.abspath(os.path.join(current_dir, f"../../embedding_{method}.png"))

    plt.savefig(save_path)
    plt.close()
    print(f"{method.upper()} の可視化画像を保存しました: {save_path}")
