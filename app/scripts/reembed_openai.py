# scripts/reembed_openai.py
"""
Re-embed all Conversation rows with OpenAI embeddings (e.g., text-embedding-3-small, 1536-dim).
- 既存がSBERT(384)などでも、OpenAIに統一して上書き保存します
- 既に1536次元のものは、--force が無い限りスキップ
- バッチコミット、レート制御、エラースキップ付き

使い方:
  # 仮実行（更新せず統計だけ）
  python -m scripts.reembed_openai --dry-run

  # 実行（埋め込みが1536以外の行のみ更新）
  python -m scripts.reembed_openai

  # 全件強制再計算
  python -m scripts.reembed_openai --force

環境変数:
  OPENAI_API_KEY (必須)
  EMBED_MODEL=text-embedding-3-small (任意。features.get_openai_embedding側で参照)
"""
import os
import sys
import time
import math
import argparse
import pickle
import traceback

import numpy as np
from sqlalchemy.orm import Session

# your project modules
from app.database import SessionLocal
from app import models
from app.features import get_openai_embedding  # bytes (pickle of np.float32 array)

TARGET_DIM = 1536  # text-embedding-3-small の次元。model変更時は合わせる

def embedding_dim_from_bytes(b: bytes | bytearray | None) -> int | None:
    if not b:
        return None
    try:
        vec = pickle.loads(b)
        arr = np.asarray(vec, dtype=np.float32).ravel()
        return int(arr.size)
    except Exception:
        return None

def reembed_all(dry_run: bool = False, force: bool = False, batch_size: int = 100, sleep_sec: float = 0.1):
    db: Session = SessionLocal()
    q = db.query(models.Conversation)
    total = q.count()
    print(f"[reembed] total rows = {total}")

    updated = 0
    scanned = 0
    errors = 0

    # 逐次バッチで処理（主キー昇順）
    last_id = 0
    while True:
        rows = (
            db.query(models.Conversation)
            .filter(models.Conversation.id > last_id)
            .order_by(models.Conversation.id.asc())
            .limit(batch_size)
            .all()
        )
        if not rows:
            break

        for r in rows:
            scanned += 1
            last_id = r.id

            msg = (r.message or "").strip()
            if not msg:
                # 無内容はスキップ
                continue

            cur_dim = embedding_dim_from_bytes(r.embedding)
            need_update = force or (cur_dim is None) or (cur_dim != TARGET_DIM)

            if not need_update:
                continue

            if dry_run:
                print(f"[dry-run] id={r.id} dim={cur_dim} -> will update to {TARGET_DIM}")
                continue

            # 実処理
            try:
                emb_bytes = get_openai_embedding(msg)  # bytes (pickle化済み想定)
                # 念のため検査
                dim = embedding_dim_from_bytes(emb_bytes)
                if dim != TARGET_DIM:
                    print(f"[warn] id={r.id} got dim={dim}, expected {TARGET_DIM}. Skipping.")
                    continue

                r.embedding = emb_bytes
                updated += 1

                # 軽いレート制御（OpenAI側への優しさ）
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

            except Exception as e:
                errors += 1
                print(f"[error] id={r.id} {e}\n{traceback.format_exc()}")

        # バッチコミット
        if not dry_run:
            db.commit()
            print(f"[commit] scanned={scanned}/{total}, updated={updated}, errors={errors}")

    db.close()
    print(f"[done] scanned={scanned}, updated={updated}, errors={errors}, dry_run={dry_run}, force={force}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="実際には更新しないで対象件数だけ確認")
    parser.add_argument("--force", action="store_true", help="既に1536でも再計算して上書き")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.1, help="OpenAI呼び出し間の待機秒。0で無効化")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)

    reembed_all(dry_run=args.dry_run, force=args.force, batch_size=args.batch_size, sleep_sec=args.sleep)

if __name__ == "__main__":
    main()
