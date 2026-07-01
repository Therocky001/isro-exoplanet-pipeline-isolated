import argparse
import csv
from pathlib import Path


VALID_LABELS = {"planet", "eclipsing_binary", "blend_noise"}


def parse_args():
    parser = argparse.ArgumentParser(description="Label rows in the exoplanet annotation queue.")
    parser.add_argument("--manifest", default="output/labeling_queue.csv", help="Path to the labeling queue CSV.")
    parser.add_argument("--tic-id", required=True, help="TIC identifier to label, e.g. TIC 80423805.")
    parser.add_argument("--label", required=True, choices=sorted(VALID_LABELS), help="Class label.")
    parser.add_argument("--notes", default="", help="Optional reviewer notes.")
    return parser.parse_args()


def label_target(manifest_path: str, tic_id: str, label: str, notes: str = ""):
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_file.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = rows[0].keys() if rows else ["tic_id", "rad", "mass", "teff", "tmag", "logg", "label", "review_status", "notes"]

    updated = False
    for row in rows:
        if row.get("tic_id") == tic_id:
            row["label"] = label
            row["review_status"] = "labeled"
            row["notes"] = notes
            updated = True
            break

    if not updated:
        raise ValueError(f"TIC ID not found in manifest: {tic_id}")

    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {tic_id} -> {label}")


def main():
    args = parse_args()
    label_target(args.manifest, args.tic_id, args.label, args.notes)


if __name__ == "__main__":
    main()