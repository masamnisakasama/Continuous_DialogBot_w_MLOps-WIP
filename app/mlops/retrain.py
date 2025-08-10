"""
from fastapi import APIRouter, BackgroundTasks
from sqlalchemy.orm import Session
from app import crud, database, features
import logging

router = APIRouter()

# ロガー設定
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 再学習タスク本体
def retrain_model():
    logger.info("🔁 [retrain_model] 再学習を開始します")
    db: Session = database.SessionLocal()
    try:
        conversations = crud.get_all_conversations(db)
        if not conversations:
            logger.warning("⚠️ 再学習対象の会話データが存在しません")
            return

        # t-SNEとPCAでの可視化を行う
        features.visualize_embeddings(conversations, method="tsne")
        logger.info("✅ TSNEによる可視化画像の生成が完了しました")

        features.visualize_embeddings(conversations, method="pca")
        logger.info("✅ PCAによる可視化画像の生成が完了しました")

        # 将来的にモデル再学習処理をここに追加可能

    except Exception as e:
        logger.error(f" 再学習中にエラーが発生しました: {e}")
    finally:
        db.close()
        logger.info("🔚 [retrain_model] 再学習処理が終了しました")

# APIエンドポイント
@router.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    
    会話データの再学習（再可視化）を非同期で実行するエンドポイント。
    
    background_tasks.add_task(retrain_model)
    logger.info("再学習タスクをバックグラウンドでキューに追加しました")
    return {"message": "再学習処理をバックグラウンドで開始しました"}
"""

# app/mlops/retrain_api.py
from fastapi import APIRouter
from typing import Dict, Any
import traceback

router = APIRouter()

@router.post("/mlops/retrain")
def retrain_endpoint() -> Dict[str, Any]:
    """
    再学習のAPIトリガー。内部で app.mlops.retrain.retrain_model() を呼びます。
    """
    try:
        # 学習ロジック本体（あなたの既存ファイル）を遅延importして循環を回避
        from app.mlops.retrain import retrain_model
    except Exception as e:
        return {"status": "error", "detail": "retrain_model が見つかりません。", "import_error": str(e)}

    try:
        result = retrain_model()  # 必要なら引数を追加
        return {"status": "ok", "result": result if result is not None else "retrain finished"}
    except Exception as e:
        return {"status": "error", "detail": str(e), "traceback": traceback.format_exc()}
