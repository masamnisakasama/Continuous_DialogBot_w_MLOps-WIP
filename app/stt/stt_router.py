"""
# app/stt/stt_router.py
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from statistics import mean, median

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Whisper（CPU固定・軽量フォールバックは whisper_utils 側）
from app.stt.whisper_utils import (
    transcribe_with_segments,
    get_model_name,
)

router = APIRouter()

# =========================
# 設定（必要なら調整）
# =========================
# ffprobe で duration が取れない時に、WAV に一時変換して再計測するか
ENABLE_WAV_DURATION_FALLBACK = True
# 変換時のサンプルレート/チャンネル（早くて十分な値）
WAV_AR = "16000"
WAV_AC = "1"
# 1KB 未満の入力は異常として弾く
MIN_BYTES = 1024


# =========================
# 補助関数
# =========================
def _probe_duration_sec(path: str) -> float:
    ffprobeで音声長を秒で取得。失敗時は0.
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return max(0.0, float(out))
    except Exception:
        return 0.0


def _probe_duration_via_wav(path: str) -> float:
    WAV に一時変換して duration を再計測。失敗時は0。
    wav_tmp = path + ".dur.wav"
    try:
        subprocess.check_call(
            ["ffmpeg", "-y", "-i", path, "-ac", WAV_AC, "-ar", WAV_AR, wav_tmp],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        return _probe_duration_sec(wav_tmp)
    except Exception:
        return 0.0
    finally:
        try:
            os.remove(wav_tmp)
        except Exception:
            pass


def _count_words_like(text: str) -> int:
    
    日本語は空白で分かれないので、ざっくり：
    - 空白で分割できるなら単純にトークン数
    - できないなら空白除去後の文字数を「擬似ワード」として使う
    
    tokens = [t for t in text.strip().split() if t]
    if tokens:
        return len(tokens)
    import re
    cleaned = re.sub(r"\s+", "", text)
    return len(cleaned)


def _metrics_from_segments(segments: list[dict], total_dur: float, full_text: str) -> dict:
    
    Whisperの segments（各 {start, end, text}）から軽量メトリクスを計算
    total_dur は 0 でも受け取るが、その場合は速度系が0に寄る
    
    if not segments:
        return {
            "speech_rate_wpm": 0.0,
            "speech_rate_cps": 0.0,
            "pause_ratio": 0.0,
            "num_pauses": 0,
            "avg_pause_sec": 0.0,
            "median_pause_sec": 0.0,
            "voiced_time_sec": 0.0,
            "utterance_density": 0.0,
            "avg_segment_sec": 0.0,
            "num_segments": 0,
        }

    seg_durations = [(s.get("end", 0.0) - s.get("start", 0.0)) for s in segments]
    seg_durations = [max(0.0, d) for d in seg_durations]
    voiced_time = sum(seg_durations)

    # ポーズ（隣接セグメントの隙間）
    pauses = []
    for i in range(1, len(segments)):
        prev_end = segments[i - 1].get("end", 0.0)
        next_start = segments[i].get("start", 0.0)
        gap = max(0.0, next_start - prev_end)
        if gap > 0.0:
            pauses.append(gap)

    total_pause = sum(pauses)
    pause_ratio = (total_pause / total_dur) if total_dur > 0 else 0.0

    # 速度
    import re
    pseudo_words = _count_words_like(full_text)
    cps = (len(re.sub(r"\s+", "", full_text)) / total_dur) if total_dur > 0 else 0.0
    wpm = ((pseudo_words / total_dur) * 60.0) if total_dur > 0 else 0.0

    return {
        "speech_rate_wpm": wpm,
        "speech_rate_cps": cps,
        "pause_ratio": min(1.0, pause_ratio) if pause_ratio > 0 else 0.0,
        "num_pauses": len(pauses),
        "avg_pause_sec": float(mean(pauses)) if pauses else 0.0,
        "median_pause_sec": float(median(pauses)) if pauses else 0.0,
        "voiced_time_sec": voiced_time,
        "utterance_density": (voiced_time / total_dur) if total_dur > 0 else 0.0,
        "avg_segment_sec": float(mean(seg_durations)) if seg_durations else 0.0,
        "num_segments": len(segments),
    }


# =========================
# ルート
# =========================
@router.get("/stt/model")
async def stt_model():
    ロード済みの Whisper モデル名を返す（初回実行後に確定）。
    name = get_model_name()
    return {"model": name or "loading_on_first_use"}


@router.post("/stt-full/")
async def analyze_audio(file: UploadFile = File(...)):
    
    軽量メトリクス付きのSTT（堅牢化版）:
      1) ffprobe で duration 計測
      2) 0 のときは WAV 変換で再計測（有効なら）
      3) それでも 0 のときは Whisper セグメントの span から推定
      4) segments から pause/速度などのメトリクス算出
    
    # 一時ファイルを安全に作成（拡張子は元に合わせる。無ければ webm）
    suffix = Path(file.filename).suffix or ".webm"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="stt_")
    os.close(fd)

    try:
        # 中身を書き出し
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 0バイト/極端に小さい入力は弾く
        try:
            if os.path.getsize(tmp_path) < MIN_BYTES:
                raise HTTPException(status_code=400, detail="Audio too small (<1KB). Please record at least 2 seconds.")
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Uploaded file missing.")

        # 1) ffprobe で測る
        duration_sec = _probe_duration_sec(tmp_path)

        # 2) Whisper で text+segments 取得（軽量設定）
        lang = "ja"   # 自動検出にしたいなら None
        result = transcribe_with_segments(tmp_path, language=lang)
        text = (result.get("text") or "").strip()
        segments = result.get("segments") or []

        # 3) duration が 0 の場合のフォールバック
        if (not duration_sec or duration_sec <= 0):
            # 3a) WAV 変換で再計測（有効な場合のみ）
            if ENABLE_WAV_DURATION_FALLBACK:
                wav_dur = _probe_duration_via_wav(tmp_path)
                if wav_dur > 0:
                    duration_sec = wav_dur

            # 3b) まだ 0 のときは segments から推定（最終手段）
            if (not duration_sec or duration_sec <= 0) and segments:
                starts = [s.get("start", 0.0) for s in segments]
                ends = [s.get("end", 0.0) for s in segments]
                if starts and ends:
                    seg_span = max(0.0, (max(ends) - min(starts)))
                    duration_sec = seg_span if seg_span > 0 else 0.0

        # 4) メトリクス計算
        audio_metrics = _metrics_from_segments(segments, duration_sec, text)

        return JSONResponse(
            {
                "text": text,
                "model": get_model_name(),
                "language": lang,
                "duration_sec": duration_sec,
                "audio_metrics": audio_metrics,
                # デバッグ用に必要なら "segments": segments,
            }
        )

    except HTTPException:
        # そのまま再raise
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")
    finally:
        # 一時ファイル掃除
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# 互換：以前のエンドポイントを残す（不要なら削除可）
@router.post("/analyze/audio")
async def analyze_audio_compat(file: UploadFile = File(...)):
    return await analyze_audio(file)
"""

# app/stt/stt_router.py
from __future__ import annotations

import os, shutil, tempfile, subprocess
from pathlib import Path
from statistics import mean, median

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

# Whisperユーティリティ（CPU固定・軽量フォールバック・segments返却）
from app.stt.whisper_utils import transcribe_with_segments, get_model_name

router = APIRouter()

# -------- 設定 --------
ENABLE_WAV_DURATION_FALLBACK = True
WAV_AR = "16000"
WAV_AC = "1"
MIN_BYTES = 1024

# -------- 補助 --------
def _probe_duration_sec(path: str) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", path],
            stderr=subprocess.STDOUT, text=True
        ).strip()
        return max(0.0, float(out))
    except Exception:
        return 0.0

def _probe_duration_via_wav(path: str) -> float:
    wav_tmp = path + ".dur.wav"
    try:
        subprocess.check_call(
            ["ffmpeg","-y","-i",path,"-ac",WAV_AC,"-ar",WAV_AR,wav_tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        return _probe_duration_sec(wav_tmp)
    except Exception:
        return 0.0
    finally:
        try: os.remove(wav_tmp)
        except Exception: pass

def _count_words_like(text: str) -> int:
    toks = [t for t in text.strip().split() if t]
    if toks: return len(toks)
    import re
    return len(re.sub(r"\s+","",text))

def _metrics_from_segments(segments: list[dict], total_dur: float, full_text: str) -> dict:
    if not segments:
        return {"speech_rate_wpm":0.0,"speech_rate_cps":0.0,"pause_ratio":0.0,"num_pauses":0,
                "avg_pause_sec":0.0,"median_pause_sec":0.0,"voiced_time_sec":0.0,
                "utterance_density":0.0,"avg_segment_sec":0.0,"num_segments":0}

    seg_durs = [max(0.0, s.get("end",0.0)-s.get("start",0.0)) for s in segments]
    voiced = sum(seg_durs)

    pauses=[]
    for i in range(1,len(segments)):
        gap = max(0.0, segments[i].get("start",0.0)-segments[i-1].get("end",0.0))
        if gap>0.0: pauses.append(gap)

    total_pause = sum(pauses)
    pause_ratio = (total_pause/total_dur) if total_dur>0 else 0.0

    import re
    cps = (len(re.sub(r"\s+","",full_text))/total_dur) if total_dur>0 else 0.0
    wpm = (((_count_words_like(full_text))/total_dur)*60.0) if total_dur>0 else 0.0

    return {
        "speech_rate_wpm": wpm,
        "speech_rate_cps": cps,
        "pause_ratio": min(1.0,pause_ratio) if pause_ratio>0 else 0.0,
        "num_pauses": len(pauses),
        "avg_pause_sec": float(mean(pauses)) if pauses else 0.0,
        "median_pause_sec": float(median(pauses)) if pauses else 0.0,
        "voiced_time_sec": voiced,
        "utterance_density": (voiced/total_dur) if total_dur>0 else 0.0,
        "avg_segment_sec": float(mean(seg_durs)) if seg_durs else 0.0,
        "num_segments": len(segments),
    }

def _make_speaking_advice(m: dict) -> list[str]:
    tips=[]
    cps=m.get("speech_rate_cps",0.0); pr=m.get("pause_ratio",0.0); den=m.get("utterance_density",0.0)
    if cps<4.0: tips.append("少しゆっくり目です。重要語に軽い抑揚を付けると伝わりやすいです。")
    elif cps>8.0: tips.append("やや速め。文末で0.2秒ほど間を置くと聞き取りやすくなります。")
    else: tips.append("話速はちょうど良いです。今のテンポを維持しましょう。")
    if pr<0.05: tips.append("ポーズが少なめ。フレーズの切れ目で小休止を入れると理解が深まります。")
    elif pr>0.2: tips.append("ポーズが多め。言い切りまで意識して一息で話すと流暢さが上がります。")
    if den>0.9: tips.append("発話密度が高め。要点毎に短く区切ると聞き手が追いやすいです。")
    elif den<0.5: tips.append("発話密度が低め。1文は短く、主語→結論の順でキレ良く。")
    return tips or ["全体として自然なペースです。大きな改善点は見当たりません。"]

# -------- ルート --------
@router.get("/stt/model")
async def stt_model():
    return {"model": get_model_name() or "loading_on_first_use"}

@router.post("/stt-full/")
async def stt_full(file: UploadFile = File(...), detail: bool = Query(False)):
    suffix = Path(file.filename).suffix or ".webm"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="stt_"); os.close(fd)
    try:
        with open(tmp_path,"wb") as f:
            shutil.copyfileobj(file.file, f)

        # サイズガード
        if os.path.getsize(tmp_path) < MIN_BYTES:
            raise HTTPException(status_code=400, detail="Audio too small (<1KB). Please record at least 2 seconds.")

        # duration 1st
        duration_sec = _probe_duration_sec(tmp_path)

        # Whisper（segments取得）
        lang="ja"  # 自動検出にしたいなら None
        result = transcribe_with_segments(tmp_path, language=lang)
        text = (result.get("text") or "").strip()
        segments = result.get("segments") or []

        # duration fallback
        if not duration_sec or duration_sec<=0:
            if ENABLE_WAV_DURATION_FALLBACK:
                wav_dur = _probe_duration_via_wav(tmp_path)
                if wav_dur>0: duration_sec = wav_dur
        if (not duration_sec or duration_sec<=0) and segments:
            starts=[s.get("start",0.0) for s in segments]; ends=[s.get("end",0.0) for s in segments]
            if starts and ends: duration_sec = max(0.0, (max(ends)-min(starts)))

        # metrics
        audio_metrics = _metrics_from_segments(segments, duration_sec, text)
        advice = _make_speaking_advice(audio_metrics)

        return JSONResponse({
            "text": text,
            "model": get_model_name(),
            "language": lang,
            "duration_sec": duration_sec,
            "audio_metrics": audio_metrics,
            "advice": advice
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

# ← これが「互換」ルート。フロントが /analyze/audio を叩いてもOKになる
@router.post("/analyze/audio")
async def analyze_audio_compat(file: UploadFile = File(...), detail: bool = Query(False)):
    return await stt_full(file=file, detail=detail)
