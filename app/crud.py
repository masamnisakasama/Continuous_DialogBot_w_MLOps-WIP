from typing import Optional, List
from sqlalchemy.orm import Session
import pickle
import numpy as np

from app import models, schemas

"""
def create_conversation(
    db: Session,
    conv: schemas.ConversationCreate,
    style: Optional[str],
    embedding: bytes,
    sentiment: Optional[str] = None,
):
    obj = models.Conversation(
        user=conv.user,
        message=conv.message,
        style=style,
        embedding=embedding,   # BLOB
        sentiment=sentiment,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj
"""

def create_conversation(
    db: Session,
    conv: schemas.ConversationCreate,
    style: Optional[str],
    embedding: bytes,
    sentiment: Optional[str] = None,
):
    # ここ：コンストラクタでは sentiment を渡さない（存在しないとTypeError）
    obj = models.Conversation(
        user=conv.user,
        message=conv.message,
        style=style,
        embedding=embedding,
    )

    # モデルに sentiment カラムがあるプロジェクトなら代入、無ければスキップ
    if sentiment is not None and hasattr(obj, "sentiment"):
        obj.sentiment = sentiment

    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def topk_similar(
    db: Session,
    query_emb: bytes | np.ndarray,
    top_k: int = 5,
    exclude_id: Optional[int] = None,
) -> List[dict]:
    if isinstance(query_emb, (bytes, bytearray)):
        try:
            qv = pickle.loads(query_emb)
        except Exception:
            return []
    else:
        qv = np.asarray(query_emb, dtype=np.float32).ravel()
    qv = np.asarray(qv, dtype=np.float32).ravel()
    qdim = qv.shape[0]

    rows = db.query(models.Conversation).all()
    scored = []
    for r in rows:
        if exclude_id and r.id == exclude_id:
            continue
        if not r.embedding:
            continue
        try:
            emb = pickle.loads(r.embedding)
            emb = np.asarray(emb, dtype=np.float32).ravel()
        except Exception:
            continue

        if emb.shape[0] != qdim:
            continue

        sim = _cosine_sim(qv, emb)
        scored.append((sim, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "id": r.id,
            "user": getattr(r, "user", None),
            "message": r.message,
            "similarity": float(sim),
        }
        for sim, r in scored[:top_k]
    ]
