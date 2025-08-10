"""
from fastapi import FastAPI, Depends, Query, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app import database, models, crud, features, schemas
from dotenv import load_dotenv
from app.mlops.retrain import router as retrain_router
import os
import whisper
import tempfile
from transformers import pipeline
import librosa
import numpy as np

load_dotenv()
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

origins = [ "http://localhost:3000",
    "http://localhost:8015",
    "http://127.0.0.1:8015",]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 感情分析pipeline初期化
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="jarvisx17/japanese-sentiment-analysis",
    tokenizer="jarvisx17/japanese-sentiment-analysis"
)

print(sentiment_analyzer("最悪"))
print(sentiment_analyzer("嬉しい"))
print(sentiment_analyzer("悲しい"))
print(sentiment_analyzer("これはテストです"))

def analyze_sentiment(text: str) -> dict:
    results = sentiment_analyzer(text)
    print(f"Sentiment results: {results}")
    print("Sentiment raw results:", results) 
    return {
        "label": results[0]["label"],
        "score": results[0]["score"],
    }

whisper_model = whisper.load_model("base")

@app.post("/conversations/", response_model=schemas.Conversation)
def create_conversation(
    conv_create: schemas.ConversationCreate, 
    db: Session = Depends(get_db)
):
    embedding = features.get_embedding(conv_create.message)
    analysis = features.classify_dialogue_style(conv_create.message)
    sentiment_label = analysis.get("emotion", None)
    db_conv = crud.create_conversation(
        db=db, 
        conv=conv_create, 
        style=analysis.get("style"), 
        embedding=embedding, 
        sentiment=sentiment_label
    )
    return db_conv

@app.get("/recommendations/with-explanation")
def get_recommendations_with_explanation(
    query: str,
    exclude_id: int = None,  
    db: Session = Depends(get_db)
):
    all_convs = crud.get_all_conversations(db)
    results = features.recommend_similar_conversations(query, all_convs, explain=True)

    response = []
    for conv, sim, explanation, explanation_text in results:
        if exclude_id and conv.id == exclude_id:
            continue  
        response.append({
            "id": conv.id,
            "message": conv.message,
            "similarity": round(sim, 4),
            "top_dimensions": explanation,
            "explanation_text": explanation_text,
        })
    return JSONResponse(content=response)

@app.get("/recommendations/", response_model=list[schemas.Conversation])
def get_recommendations(query: str, db: Session = Depends(get_db)):
    all_convs = crud.get_all_conversations(db)
    top_convs = features.recommend_similar_conversations(query, all_convs)
    return top_convs

@app.get("/visualize/image")
def get_visualization_image(method: str = Query("tsne", regex="^(tsne|pca)$")):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.abspath(os.path.join(current_dir, f"../../embedding_{method}.png"))

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")

    return FileResponse(filepath, media_type="image/png")

@app.post("/visualize/generate")
def generate_visualizations(db: Session = Depends(get_db)):
    conversations = crud.get_all_conversations(db)
    if not conversations:
        return {"error": "会話データがありません"}

    try:
        features.visualize_embeddings(conversations, method="tsne")
        features.visualize_embeddings(conversations, method="pca")
        return {"message": "可視化画像の生成が完了しました"}
    except Exception as e:
        return {"error": str(e)}

後で消そう
@app.post("/stt/")
async def stt_and_save(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        return {"message": "STT endpoint is alive"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/stt/")
async def stt_and_save(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print(f"保存した音声ファイルサイズ: {os.path.getsize(tmp_path)} バイト")

        # Whisperで音声認識
        result = whisper_model.transcribe(tmp_path, language="ja")
        message = result["text"]
        print(f"Whisper認識結果: {message!r}")

        # ===== 追加: 音声分析（話速・抑揚） =====
        y, sr = librosa.load(tmp_path, sr=None)
        duration_sec = librosa.get_duration(y=y, sr=sr)
        speaking_rate = len(message) / duration_sec if duration_sec > 0 else 0

        volume = librosa.feature.rms(y=y)[0]
        intonation_var = float(np.std(volume))

        # 感情分析
        sentiment = analyze_sentiment(message)
        print(f"感情分析結果: {sentiment}")

        # 特徴量抽出
        embedding = features.get_embedding(message)
        style = features.classify_dialogue_style(message).get("style")

        print(f"embedding: {embedding}")
        print(f"style: {style}")

        # DBに保存
        conv_create = schemas.ConversationCreate(user="anonymous", message=message)
        db_conv = crud.create_conversation(
            db=db,
            conv=conv_create,
            style=style,
            embedding=embedding,
            sentiment=sentiment["label"],
        )

        #  類似会話検索（自分自身は除外） 
        all_convs = crud.get_all_conversations(db)
        results = features.recommend_similar_conversations(message, all_convs, explain=True)
        similar_list = []
        for conv, sim, explanation, explanation_text in results:
            if conv.id == db_conv.id:  # 自分自身の会話を表示
                continue
            similar_list.append({
                "id": conv.id,
                "message": conv.message,
                "similarity": round(sim, 4),
                "top_dimensions": explanation,
                "explanation_text": explanation_text,
            })
    

        return {
            "text": message,
            "sentiment": sentiment,
            "conversation_id": db_conv.id,
            "speaking_rate": speaking_rate,
            "intonation_variation": intonation_var,
            "similar_conversations": similar_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT処理中にエラーが発生しました: {str(e)}")
app.include_router(retrain_router, prefix="/mlops")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8015, reload=True)
"""

# app/main.py
import logging

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# プロジェクト内モジュール
from app.database import get_db
from app import schemas, crud, features

# ルーター群
from app.stt.stt_router import router as stt_router           # /analyze/audio など
from app.mlops.retrain_api import router as retrain_router    # /mlops/retrain
from app.mlops.drift_router import router as drift_router     # /drift/rebase, /drift/status

# ===== アプリ生成（ここだけで作る） =====
app = FastAPI(title="Dialog Bot API")

# ===== ミドルウェア =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # 本番はフロントのオリジンに絞ってください
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ルーター登録 =====
app.include_router(stt_router)
app.include_router(retrain_router)
app.include_router(drift_router)

# ===== 単発エンドポイント（必要最小限） =====
class UpsertRequest(BaseModel):
    text: str
    speaker: str | None = None
    top_k: int = 5

class UpsertResponse(BaseModel):
    id: int
    similar: list[dict]  # {id, user, message, similarity}

@app.post("/conversations/upsert_and_search", response_model=UpsertResponse)
def upsert_and_search(body: UpsertRequest, db: Session = Depends(get_db)):
    """
    OpenAI Embedding を一回だけ実行 → 保存 → 既存データとの類似Top-kを返す。
    """
    # 1) Embedding（OpenAI）
    emb_bytes = features.get_openai_embedding(body.text)

    # 2) 保存
    conv_create = schemas.ConversationCreate(
        user=body.speaker or "user",
        message=body.text
    )
    saved = crud.create_conversation(
        db=db,
        conv=conv_create,
        style=None,
        embedding=emb_bytes,
        sentiment=None,  # モデルに列が無い場合も安全に動くようcrud側でhasattrチェック済み
    )

    # 3) 類似検索（自分は除外）
    sims = crud.topk_similar(
        db,
        query_emb=emb_bytes,
        top_k=body.top_k,
        exclude_id=saved.id
    )
    return UpsertResponse(id=saved.id, similar=sims)

@app.get("/health")
def health():
    return {"status": "ok"}

# ===== ログ設定（任意） =====
logging.basicConfig(level=logging.INFO)

# ===== 起動時デバッグ：ルート一覧を出力（不要なら削除OK） =====
for r in app.routes:
    try:
        print("ROUTE", getattr(r, "methods", None), getattr(r, "path", None))
    except Exception:
        pass

# ===== 直起動用（uvicornから呼ぶなら不要） =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8015, reload=True)
