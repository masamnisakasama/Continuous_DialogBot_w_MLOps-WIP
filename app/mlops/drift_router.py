"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import numpy as np, pickle, json, os, time
from datetime import datetime, timedelta

from app.database import get_db
from app import models, features

router = APIRouter()

REF_PATH = os.getenv("DRIFT_REF_PATH", "./drift_ref.json")

def _fetch_rows(db: Session, since: datetime) -> List[models.Conversation]:
    return (
        db.query(models.Conversation)
        .filter(models.Conversation.created_at >= since)  # created_at が無いなら id範囲でもOK
        .all()
    )

def _safe_vec(b: bytes) -> np.ndarray | None:
    if not b: return None
    try:
        v = pickle.loads(b)
        a = np.asarray(v, dtype=np.float32).ravel()
        return a if a.size > 0 else None
    except Exception:
        return None

def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    #Population Stability Index
    # ヒストグラムのbin境界を揃える
    r_hist, edges = np.histogram(ref, bins=bins)
    c_hist, _ = np.histogram(cur, bins=edges)
    # 割合に変換（ゼロ除算防止のためclip）
    r_p = np.clip(r_hist / max(1, r_hist.sum()), 1e-6, 1)
    c_p = np.clip(c_hist / max(1, c_hist.sum()), 1e-6, 1)
    # PSI計算
    return float(np.sum((r_p - c_p) * np.log(r_p / c_p)))

def _cos_center_shift(ref_mat: np.ndarray, cur_mat: np.ndarray) -> float:
    if len(ref_mat)==0 or len(cur_mat)==0: return 0.0
    rc = ref_mat.mean(axis=0); cc = cur_mat.mean(axis=0)
    na = np.linalg.norm(rc); nb = np.linalg.norm(cc)
    if na==0 or nb==0: return 0.0
    return float(1 - np.dot(rc, cc) / (na * nb))  # 0=同一, 1=直交

def _text_stats(rows: List[models.Conversation]) -> Dict[str, np.ndarray]:
    lens = []
    tech_ratio = []
    sent = []
    for r in rows:
        t = (r.message or "").strip()
        if not t: continue
        lens.append(len(t))
        # ↓必要なら簡易な専門語辞書で置換（今は疑似：数字/英字比率など）
        tech_ratio.append(sum(ch.isalnum() for ch in t)/max(1,len(t)))
        # 感情は既存のfeaturesがあれば利用、無ければ0に
        sent.append(0.0)
    return {
        "len": np.asarray(lens, dtype=np.float32),
        "tech": np.asarray(tech_ratio, dtype=np.float32),
        "sent": np.asarray(sent, dtype=np.float32),
    }

def _emb_matrix(rows: List[models.Conversation]) -> np.ndarray:
    emb = []
    for r in rows:
        v = _safe_vec(r.embedding)
        if v is not None:
            emb.append(v)
    return np.asarray(emb, dtype=np.float32)

def _load_ref() -> Dict[str, Any] | None:
    if os.path.exists(REF_PATH):
        with open(REF_PATH,"r") as f:
            return json.load(f)
    return None

def _save_ref(payload: Dict[str, Any]) -> None:
    with open(REF_PATH,"w") as f:
        json.dump(payload, f)

@router.post("/drift/rebase")
def rebase_reference(days:int=30, db: Session = Depends(get_db)):
    since = datetime.utcnow() - timedelta(days=days)
    rows = _fetch_rows(db, since)
    emb = _emb_matrix(rows)
    stats = _text_stats(rows)
    ref = {
        "ts": time.time(),
        "n": len(rows),
        "emb_center": emb.mean(axis=0).tolist() if len(emb)>0 else [],
        "len_mean": float(stats["len"].mean()) if stats["len"].size>0 else 0.0,
        "len_std":  float(stats["len"].std())  if stats["len"].size>0 else 1.0,
        "tech_mean": float(stats["tech"].mean()) if stats["tech"].size>0 else 0.0,
        "tech_std":  float(stats["tech"].std())  if stats["tech"].size>0 else 1.0,
    }
    _save_ref(ref)
    return {"status":"ok","reference_size":ref["n"]}

@router.get("/drift/status")
def drift_status(hours:int=72, db: Session = Depends(get_db)):
    ref = _load_ref()
    if not ref:
        return {"status":"no_reference","hint":"POST /drift/rebase を先に実行してください"}
    since = datetime.utcnow() - timedelta(hours=hours)
    rows = _fetch_rows(db, since)
    emb = _emb_matrix(rows)
    stats = _text_stats(rows)

    # PSI（長さ・“技術っぽさ”の簡易比率）
    psi_len = _psi(
        np.random.normal(ref["len_mean"], max(ref["len_std"],1e-6), size=5000),
        stats["len"] if stats["len"].size>0 else np.array([ref["len_mean"]]),
    )
    psi_tech = _psi(
        np.random.normal(ref["tech_mean"], max(ref["tech_std"],1e-6), size=5000),
        stats["tech"] if stats["tech"].size>0 else np.array([ref["tech_mean"]]),
    )

    # 中心コサイン距離
    ref_center = np.asarray(ref["emb_center"], dtype=np.float32)
    center_shift = _cos_center_shift(
        ref_center.reshape(1,-1) if ref_center.size>0 else np.zeros((1,1536),np.float32),
        emb if emb.size>0 else np.zeros((1,1536),np.float32),
    )

    flags = {
        "psi_len": psi_len > 0.2,
        "psi_tech": psi_tech > 0.2,
        "center_shift": center_shift > 0.05  # 初期値。運用で調整
    }
    overall = any(flags.values())

    return {
        "status": "drifting" if overall else "stable",
        "n_recent": len(rows),
        "metrics": {
            "psi_len": psi_len,
            "psi_tech": psi_tech,
            "center_shift": center_shift,
        },
        "flags": flags,
        "reference_ts": ref["ts"],
    }
"""
# app/mlops/drift_router.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pickle, json, os, time, re

from app.database import get_db
from app import models

router = APIRouter()

# 参照統計の保存先（JSON）
REF_PATH = os.getenv("DRIFT_REF_PATH", "./drift_ref.json")

# ----------------- ユーティリティ -----------------

def _load_ref() -> Optional[Dict[str, Any]]:
    if os.path.exists(REF_PATH):
        with open(REF_PATH, "r") as f:
            return json.load(f)
    return None

def _save_ref(payload: Dict[str, Any]) -> None:
    with open(REF_PATH, "w") as f:
        json.dump(payload, f)

def _safe_vec(b: Optional[bytes]) -> Optional[np.ndarray]:
    if not b:
        return None
    try:
        v = pickle.loads(b)
        a = np.asarray(v, dtype=np.float32).ravel()
        return a if a.size > 0 else None
    except Exception:
        return None

def _has_created_at() -> bool:
    return hasattr(models.Conversation, "created_at")

def _fetch_reference_rows(db: Session, days: int) -> List[models.Conversation]:
    if _has_created_at():
        since = datetime.utcnow() - timedelta(days=days)
        return (
            db.query(models.Conversation)
            .filter(models.Conversation.created_at >= since)
            .all()
        )
    # created_at が無い場合は全件を参照に
    return db.query(models.Conversation).all()

def _fetch_recent_rows(db: Session, hours: int, recent_n: int = 200) -> List[models.Conversation]:
    if _has_created_at():
        since = datetime.utcnow() - timedelta(hours=hours)
        return (
            db.query(models.Conversation)
            .filter(models.Conversation.created_at >= since)
            .all()
        )
    # created_at が無い場合は “最近N件” を近似として使用
    return (
        db.query(models.Conversation)
        .order_by(models.Conversation.id.desc())
        .limit(recent_n)
        .all()
    )

# ----------- 特徴抽出（数値） -----------

_LATIN_RE = re.compile(r"[A-Za-z0-9]")  # 日本語は False、英数字のみ True

def _text_stats(rows: List[models.Conversation]) -> Dict[str, np.ndarray]:
    lens, tech_ratio = [], []
    for r in rows:
        t = (r.message or "").strip()
        if not t:
            continue
        lens.append(len(t))
        latin = sum(1 for ch in t if _LATIN_RE.match(ch))
        tech_ratio.append(latin / max(1, len(t)))
    return {
        "len": np.asarray(lens, dtype=np.float32),
        "tech": np.asarray(tech_ratio, dtype=np.float32),
    }

def _emb_matrix(rows: List[models.Conversation]) -> np.ndarray:
    emb = []
    for r in rows:
        v = _safe_vec(getattr(r, "embedding", None))
        if v is not None:
            emb.append(v)
    return np.asarray(emb, dtype=np.float32)

# ----------- 指標計算 -----------

def _psi_from_hist(ref_p: np.ndarray, cur_p: np.ndarray) -> float:
    # 安全クリップ + ログPSI
    r = np.clip(ref_p, 1e-6, 1)
    c = np.clip(cur_p, 1e-6, 1)
    return float(np.sum((r - c) * np.log(r / c)))

def _cos_center_shift(ref_mat: np.ndarray, cur_mat: np.ndarray) -> float:
    """埋め込み中心のコサイン距離（0=同一, 1=直交）"""
    if ref_mat.size == 0 or cur_mat.size == 0:
        return 0.0
    rc = ref_mat.mean(axis=0)
    cc = cur_mat.mean(axis=0)
    na = np.linalg.norm(rc); nb = np.linalg.norm(cc)
    if na == 0 or nb == 0:
        return 0.0
    return float(1 - np.dot(rc, cc) / (na * nb))

def _severity_from_psi(v: float) -> str:
    if v < 0.1:
        return "low"
    if v < 0.25:
        return "moderate"
    return "high"

def _severity_from_shift(v: float) -> str:
    if v < 0.03:
        return "low"
    if v < 0.07:
        return "moderate"
    return "high"

# ----------------- API -----------------

@router.post("/drift/rebase")
def rebase_reference(days: int = 30, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    参照窓（過去 n 日 or 全件）の統計を再計算して保存。
    ヒストグラムの bin 境界（edges）と参照頻度（ref_p）も保存して、PSIの安定性を担保。
    """
    rows = _fetch_reference_rows(db, days=days)
    stats = _text_stats(rows)
    emb = _emb_matrix(rows)

    len_vals = stats["len"]; tech_vals = stats["tech"]

    # ヒストグラムの境界（参照側の分布で固定）
    len_edges = np.histogram_bin_edges(len_vals, bins=10) if len_vals.size else np.array([], dtype=np.float64)
    tech_edges = np.histogram_bin_edges(tech_vals, bins=10) if tech_vals.size else np.array([], dtype=np.float64)

    # 参照頻度（正規化）
    if len_edges.size:
        len_ref_hist, _ = np.histogram(len_vals, bins=len_edges)
        len_ref_p = (len_ref_hist / max(1, len_ref_hist.sum())).astype(float).tolist()
    else:
        len_ref_p = []

    if tech_edges.size:
        tech_ref_hist, _ = np.histogram(tech_vals, bins=tech_edges)
        tech_ref_p = (tech_ref_hist / max(1, tech_ref_hist.sum())).astype(float).tolist()
    else:
        tech_ref_p = []

    ref_payload = {
        "ts": time.time(),
        "window_days": days,
        "n_rows": len(rows),
        # 数値分布の要約
        "len_mean": float(len_vals.mean()) if len_vals.size else 0.0,
        "len_std": float(len_vals.std()) if len_vals.size else 1.0,
        "tech_mean": float(tech_vals.mean()) if tech_vals.size else 0.0,
        "tech_std": float(tech_vals.std()) if tech_vals.size else 1.0,
        # 埋め込み中心
        "emb_center": emb.mean(axis=0).tolist() if emb.size else [],
        "emb_dim": int(emb.shape[1]) if emb.ndim == 2 and emb.shape[1] else 1536,
        # PSI用：参照ヒストの境界と頻度
        "len_edges": len_edges.tolist(),
        "len_ref_p": len_ref_p,
        "tech_edges": tech_edges.tolist(),
        "tech_ref_p": tech_ref_p,
    }
    _save_ref(ref_payload)
    return {
        "status": "ok",
        "reference": {**ref_payload, "ts_iso": datetime.utcfromtimestamp(ref_payload["ts"]).isoformat() + "Z"}
    }

@router.get("/drift/status")
def drift_status(hours: int = 72, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    直近 window（デフォ72時間 or 最近N件）と参照窓の差分を計測。
    返すもの：
      - PSI（len, tech）: 参照edges＆参照頻度に基づき計算
      - 埋め込み中心コサイン距離
      - 各メトリクス severity と全体 label（stable / watch / drifting）
    """
    ref = _load_ref()
    if not ref:
        return {"status": "no_reference", "hint": "POST /drift/rebase を先に実行してください"}

    rows = _fetch_recent_rows(db, hours=hours, recent_n=200)
    MIN_N = 10  # 直近サンプルが少なすぎる場合は保留
    if len(rows) < MIN_N:
        return {"status": "insufficient_data", "n_recent": len(rows)}

    stats_cur = _text_stats(rows)
    emb_cur = _emb_matrix(rows)

    # PSI（参照のedgesとref_pを使用）
    def psi_with_saved(ref_p_key: str, edges_key: str, cur_vals: np.ndarray) -> float:
        ref_p = np.asarray(ref.get(ref_p_key, []), dtype=np.float64)
        edges = np.asarray(ref.get(edges_key, []), dtype=np.float64)
        if cur_vals.size == 0 or ref_p.size == 0 or edges.size == 0:
            return 0.0
        cur_hist, _ = np.histogram(cur_vals, bins=edges)
        cur_p = cur_hist / max(1, cur_hist.sum())
        return _psi_from_hist(ref_p, cur_p)

    psi_len = psi_with_saved("len_ref_p", "len_edges", stats_cur["len"])
    psi_tech = psi_with_saved("tech_ref_p", "tech_edges", stats_cur["tech"])

    # 埋め込み中心距離
    ref_center = np.asarray(ref.get("emb_center", []), dtype=np.float32)
    if ref_center.size == 0:
        center_shift = 0.0
    else:
        center_shift = _cos_center_shift(ref_center.reshape(1, -1), emb_cur)

    sev = {
        "psi_len": _severity_from_psi(psi_len),
        "psi_tech": _severity_from_psi(psi_tech),
        "center_shift": _severity_from_shift(center_shift),
    }
    order = {"low": 0, "moderate": 1, "high": 2}
    overall_level = max(sev.values(), key=lambda k: order[k])
    label = {"low": "stable", "moderate": "watch", "high": "drifting"}[overall_level]

    return {
        "status": label,
        "window_hours": hours,
        "n_recent": len(rows),
        "metrics": {
            "psi_len": psi_len,
            "psi_tech": psi_tech,
            "center_shift": center_shift,
        },
        "severity": sev,
        "reference": {
            "ts": ref["ts"],
            "ts_iso": datetime.utcfromtimestamp(ref["ts"]).isoformat() + "Z",
            "window_days": ref.get("window_days"),
            "n_rows": ref.get("n_rows"),
            "emb_dim": ref.get("emb_dim"),
        },
    }
