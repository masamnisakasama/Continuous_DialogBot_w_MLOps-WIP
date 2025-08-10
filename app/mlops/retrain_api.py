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
        from app.mlops.retrain import retrain_model
    except Exception as e:
        return {
            "status": "error",
            "detail": "retrain_model が見つかりません。mlops/retrain.py を確認してください。",
            "import_error": str(e),
        }

    try:
        result = retrain_model()
        return {"status": "ok", "result": result if result is not None else "retrain finished"}
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e),
            "traceback": traceback.format_exc()
        }
