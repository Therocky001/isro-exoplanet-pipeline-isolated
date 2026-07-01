import argparse

from pipeline.training import train_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train the exoplanet classifier from a labeled manifest.")
    parser.add_argument("--manifest", default="output/toi_labeled_training_manifest_fast.csv", help="Path to a CSV manifest with tic_id and label columns.")
    parser.add_argument("--checkpoint", default="output/exoplanet_classifier.pt", help="Path to save the trained checkpoint.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--use-real-lightcurves", action="store_true", help="Fetch and use live TESS light curves instead of the synthetic fast bootstrap features.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_classifier(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_real_lightcurves=args.use_real_lightcurves,
    )


if __name__ == "__main__":
    main()