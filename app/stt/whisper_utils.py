
# app/stt/whisper_utils.py
from __future__ import annotations

import io
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional, Union, List

import whisper  # pip install openai-whisper

# =========================
# 設定
# =========================
# MPS/GPUは使わずCPU固定
DEVICE: str = "cpu"

# 環境変数でモデル名を上書き可能。指定がなければ small→base→tiny の順でフォールバック
ENV_MODEL = os.getenv("WHISPER_MODEL", "").strip()
DEFAULT_CANDIDATES: List[str] = ["small", "base", "tiny"]

# 受け付ける最大アップロードサイズ（MB）
MAX_UPLOAD_MB = float(os.getenv("WHISPER_MAX_UPLOAD_MB", "50"))

# 軽量・安定寄りの推論設定（必要なら調整）
TRANSCRIBE_KW = dict(
    temperature=0.0,
    beam_size=1,
    best_of=1,
    condition_on_previous_text=False,
    word_timestamps=False,   # ← セグメントの start/end だけ使うので False のままで軽い
    fp16=False,              # CPUなので fp16 は無効
)

# =========================
# モデルの遅延ロード（スレッドセーフ）
# =========================
_model = None
_model_name: Optional[str] = None
_model_lock = threading.Lock()

def _load_model_with_fallback():
    global _model, _model_name
    if _model is not None:
        return

    candidates: List[str] = (
        [ENV_MODEL] + [m for m in DEFAULT_CANDIDATES if m != ENV_MODEL]
        if ENV_MODEL else DEFAULT_CANDIDATES
    )
    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            local = whisper.load_model(name, device=DEVICE)
            _model = local
            _model_name = name
            print(f"[whisper_utils] Loaded model '{name}' on {DEVICE}.")
            return
        except Exception as e:
            print(f"[whisper_utils] Failed to load '{name}' on {DEVICE}: {e}")
            last_err = e
            continue
    raise RuntimeError(f"Failed to load Whisper model. Tried: {candidates}. Last error: {last_err}")

def get_model_name() -> Optional[str]:
    return _model_name

def get_model():
    """必要になったタイミングでロードして返す。"""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        _load_model_with_fallback()
        return _model

# 互換性のために module 属性としても expose（古いコードが whisper_utils.model を参照しても壊れにくく）
model = None  # 初期は None。get_model() 呼び出し後に実体をセットする。

# =========================
# 入力ハンドリング
# =========================
def _filesize_mb(path: Union[str, Path]) -> float:
    p = Path(path)
    return p.stat().st_size / (1024 * 1024)

def _to_temp_audio_file(
    data: Union[bytes, io.BufferedReader, io.BytesIO, str, Path],
    suffix: str = ".wav",
) -> str:
    """file-like/bytes/パス いずれも受け取り、一時ファイルのパスを返す。"""
    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")
        if _filesize_mb(p) > MAX_UPLOAD_MB:
            raise ValueError(f"Audio file too large (> {MAX_UPLOAD_MB} MB).")
        return str(p)

    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="whisper_")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            if isinstance(data, (io.BufferedReader, io.BytesIO)):
                f.write(data.read())
            elif isinstance(data, (bytes, bytearray)):
                f.write(data)
            else:
                raise TypeError(f"Unsupported audio data type: {type(data)}")
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

    if _filesize_mb(tmp_path) > MAX_UPLOAD_MB:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise ValueError(f"Audio file too large (> {MAX_UPLOAD_MB} MB).")

    return tmp_path

# =========================
# 文字起こしAPI
# =========================
def transcribe_audio(
    file_like_or_path: Union[bytes, io.BufferedReader, io.BytesIO, str, Path],
    *,
    language: Optional[str] = "ja",
    initial_prompt: Optional[str] = None,
) -> str:
    """
    テキストのみ欲しいときの軽量ヘルパー。
    """
    global model
    mdl = get_model()
    model = mdl  # 互換 expose
    tmp = _to_temp_audio_file(file_like_or_path, suffix=".wav")
    try:
        result = mdl.transcribe(
            tmp,
            language=language,
            initial_prompt=initial_prompt or "",
            **TRANSCRIBE_KW,
        )
        return (result or {}).get("text", "").strip()
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

def transcribe_with_segments(
    file_like_or_path: Union[bytes, io.BufferedReader, io.BytesIO, str, Path],
    *,
    language: Optional[str] = "ja",
    initial_prompt: Optional[str] = None,
) -> dict:
    """
    Whisperの result をそのまま返す（text と segments を含む）。
    word_timestamps=False でも segments の start/end は入るので軽量。
    """
    global model
    mdl = get_model()
    model = mdl  # 互換 expose
    tmp = _to_temp_audio_file(file_like_or_path, suffix=".wav")
    try:
        result = mdl.transcribe(
            tmp,
            language=language,
            initial_prompt=initial_prompt or "",
            **TRANSCRIBE_KW,
        )
        return result or {}
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass
