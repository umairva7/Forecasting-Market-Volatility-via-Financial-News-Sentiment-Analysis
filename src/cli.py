"""Simple CLI to run the full pipeline."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import data_loader, preprocessing, train


def run_phase1(args) -> None:
    data_loader.build_processed_dataset(
        news_path=args.news_path,
        output_path=args.processed_path,
        period=args.period,
    )


def run_phase2(args) -> None:
    df = preprocessing.load_data(args.processed_path)
    X_num, X_text, y, scaler = preprocessing.preprocess_features(
        df,
        target_col=args.target_col,
        seq_length=args.seq_length,
        embedding_cache_path=args.embedding_cache,
        use_embedding_cache=not args.no_embedding_cache,
    )
    splits = preprocessing.create_data_splits(
        X_num,
        X_text,
        y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    preprocessing.save_splits(splits, scaler, output_dir=args.tensors_dir)


def run_train(args) -> None:
    train.train_model(
        data_dir=args.tensors_dir,
        model_output_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


def run_all(args) -> None:
    run_phase1(args)
    run_phase2(args)
    run_train(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the multimodal volatility pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase1 = subparsers.add_parser("phase1", help="Download market data and align news")
    phase1.add_argument("--news-path", default=data_loader.DEFAULT_NEWS_PATH)
    phase1.add_argument("--processed-path", default="data/processed_market_news.csv")
    phase1.add_argument("--period", default="5y")

    phase2 = subparsers.add_parser("phase2", help="Preprocess data and cache embeddings")
    phase2.add_argument("--processed-path", default="data/processed_market_news.csv")
    phase2.add_argument("--target-col", default="^GSPC_Close")
    phase2.add_argument("--seq-length", type=int, default=10)
    phase2.add_argument("--embedding-cache", default="data/embeddings/finbert_embeddings.pkl")
    phase2.add_argument("--no-embedding-cache", action="store_true")
    phase2.add_argument("--tensors-dir", default="data/tensors")
    phase2.add_argument("--train-ratio", type=float, default=0.7)
    phase2.add_argument("--val-ratio", type=float, default=0.15)

    train_parser = subparsers.add_parser("train", help="Train the Keras model")
    train_parser.add_argument("--tensors-dir", default="data/tensors")
    train_parser.add_argument("--model-path", default="models/multimodal_vol_model.keras")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=32)

    all_parser = subparsers.add_parser("all", help="Run all phases end-to-end")
    all_parser.add_argument("--news-path", default=data_loader.DEFAULT_NEWS_PATH)
    all_parser.add_argument("--processed-path", default="data/processed_market_news.csv")
    all_parser.add_argument("--period", default="5y")
    all_parser.add_argument("--target-col", default="^GSPC_Close")
    all_parser.add_argument("--seq-length", type=int, default=10)
    all_parser.add_argument("--embedding-cache", default="data/embeddings/finbert_embeddings.pkl")
    all_parser.add_argument("--no-embedding-cache", action="store_true")
    all_parser.add_argument("--tensors-dir", default="data/tensors")
    all_parser.add_argument("--train-ratio", type=float, default=0.7)
    all_parser.add_argument("--val-ratio", type=float, default=0.15)
    all_parser.add_argument("--model-path", default="models/multimodal_vol_model.keras")
    all_parser.add_argument("--epochs", type=int, default=20)
    all_parser.add_argument("--batch-size", type=int, default=32)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "phase1":
        run_phase1(args)
    elif args.command == "phase2":
        run_phase2(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "all":
        run_all(args)


if __name__ == "__main__":
    main()
