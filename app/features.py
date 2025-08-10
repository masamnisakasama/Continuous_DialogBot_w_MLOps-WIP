import librosa
import numpy as np
import os
import json
import pickle
import numpy as np
import openai
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from transformers import pipeline
import librosa
import numpy as np
import logging

# features.py の import 群の下に追加
import os
import pickle
import numpy as np
from dotenv import load_dotenv
import openai

# features.py
import os, pickle, numpy as np
from openai import OpenAI  # ← 追加（v1系の公式クライアント）
from dotenv import load_dotenv

load_dotenv()
# openai.api_key のセットは残っていても害はありません。無くてもOK。

_client = OpenAI()  # ← グローバルに1つ作って使い回し

def get_openai_embedding(text: str) -> bytes:
    """
    OpenAI Embeddings (v1 SDK) を1回呼び、
    numpy(float32) → pickle(bytes) で返す。
    """
    model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    resp = _client.embeddings.create(model=model, input=text)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    return pickle.dumps(vec)



load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_openai_embedding(text: str) -> bytes:
    """
    OpenAI Embeddings API（text-embedding-3-small など）を1回だけ叩いて
    ベクトルを取得し、pickle化して bytes で返す（DBの LargeBinary にそのまま保存できる）。
    """
    model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    try:
        resp = openai.embeddings.create(model=model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return pickle.dumps(vec)
    except Exception as e:
        # 失敗時はフォールバックとして既存の SBERT を使う（ダウンしないよう保険）
        embedding = bert_model.encode(text)
        return pickle.dumps(embedding)


def get_embedding(text: str) -> bytes:
    embedding = bert_model.encode(text)
    return pickle.dumps(embedding)

def classify_dialogue_style(text: str) -> dict:
    system_prompt = """
    あなたは会話分析の専門家です。  
    これから渡す日本語の発言について、以下の4つの属性をJSON形式で返してください。  
    - style（話し方のスタイル）：丁寧、カジュアル、フレンドリー、専門的、簡潔、抽象的など  
    - emotion（感情）：喜び、怒り、悲しみ、驚き、恐怖、嫌悪、ニュートラル、その他  
    - emotional_intensity（感情の強さ）：小さい、中くらい、大きい  
    - topic（話題）：技術、芸術、哲学、趣味、仕事、家庭、ニュース、その他

    JSON形式で返してください：
    {
        "style": "...",
        "emotion": "...",
        "emotional_intensity": "...",
        "topic": "..."
    }
    """

    user_prompt = f"発言: {text}"

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        content = response.choices[0].message.content.strip()
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {
                "style": "不明",
                "emotion": "不明",
                "emotional_intensity": "不明",
                "topic": "不明",
            }
        for key in ["style", "emotion", "emotional_intensity", "topic"]:
            if key not in result:
                result[key] = "不明"
        return result
    except Exception as e:
        print("OpenAI API 呼び出し時の例外発生:", e)
        return {
            "style": "不明",
            "emotion": "不明",
            "emotional_intensity": "不明",
            "topic": "不明",
        }

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_vector_explanation(a: np.ndarray, b: np.ndarray, top_k: int = 5) -> list:
    contribution = a * b
    top_dims = np.argsort(contribution)[-top_k:][::-1]
    return top_dims.tolist()

def generate_natural_language_explanation(query: str, target: str) -> str:
    try:
        prompt = f"""
        あなたは会話スタイル解析の専門家です。以下の2つの発言がなぜ似ていると判断できるかを日本語で説明してください。

        発言1: {query}
        発言2: {target}

        類似点、話題、感情、文体などに触れてください。
        """
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"説明の生成に失敗しました: {str(e)}"

def recommend_similar_conversations(query_text: str, conversations: list, explain: bool = False) -> list:
    query_vec = bert_model.encode(query_text)
    similarities = []

    for conv in conversations:
        if conv.embedding:
            emb = pickle.loads(conv.embedding)
            sim = cosine_similarity(query_vec, emb)
            explanation = get_vector_explanation(query_vec, emb) if explain else []
            explanation_text = None
            if explain:
                explanation_text = generate_natural_language_explanation(query_text, conv.message)
            similarities.append((conv, sim, explanation, explanation_text))
    
    logging.debug(f"類似度計算結果: {similarities}")

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_convs = similarities[:5]
    return top_convs

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

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.abspath(os.path.join(current_dir, f"../../embedding_{method}.png"))

    plt.savefig(save_path)
    plt.close()
    print(f"{method.upper()} の可視化画像を保存しました: {save_path}")



def compute_audio_metrics(audio_path: str, recognized_text: str | None = None) -> dict:
    """
    話速(文字/秒)・抑揚(簡易)・基本統計を返す
    - duration_sec: 音声長（秒）
    - speed: 文字数/秒  ※recognized_text が空なら None
    - intonation: 簡易指標（F0の変動 + エネルギー変動を合成）
    - rms_var: エネルギーの分散（大きいほど強弱の変化がある）
    - zcr_mean: ゼロ交差率（粗い発話スピードの参考値）
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # エネルギー（RMS）とゼロ交差率
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms_var = float(np.var(rms))
    zcr_mean = float(np.mean(zcr))

    # ピッチ(基本周波数)の揺らぎ
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # ≈65 Hz
            fmax=librosa.note_to_hz('C7')   # ≈2093 Hz
        )
        f0 = f0[~np.isnan(f0)]
        f0_cv = float(np.std(f0) / np.mean(f0)) if f0.size and np.mean(f0) > 0 else 0.0
    except Exception:
        f0_cv = 0.0

    # “抑揚”の簡易スコア（0〜おおよそ1台で収まりやすいように軽く圧縮）
    intonation = 0.6 * min(f0_cv, 1.0) + 0.4 * min(rms_var * 5.0, 1.0)

    # 話速（文字/秒）
    speed = None
    if recognized_text and duration > 0:
        speed = float(len(recognized_text) / duration)

    return {
        "duration_sec": round(duration, 2),
        "speed": None if speed is None else round(speed, 2),
        "intonation": round(float(intonation), 2),
        "rms_var": round(rms_var, 5),
        "zcr_mean": round(zcr_mean, 5),
    }


    