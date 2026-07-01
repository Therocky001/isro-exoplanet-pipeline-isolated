import csv
from pathlib import Path

import duckdb


def build_labeling_queue(
    filtered_index_path: str = "output/filtered_index.csv",
    output_path: str = "output/labeling_queue.csv",
    limit: int = 250,
):
    filtered_file = Path(filtered_index_path)
    if not filtered_file.exists():
        raise FileNotFoundError(f"Filtered index not found: {filtered_index_path}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(":memory:")
    query = f"""
        SELECT
            'TIC ' || ID AS tic_id,
            rad,
            mass,
            teff,
            tmag,
            logg,
            '' AS label,
            'unlabeled' AS review_status,
            '' AS notes
        FROM read_csv_auto('{filtered_file.as_posix()}')
        ORDER BY tmag ASC NULLS LAST, logg DESC NULLS LAST
        LIMIT {int(limit)}
    """

    rows = conn.execute(query).fetchall()
    headers = ["tic_id", "rad", "mass", "teff", "tmag", "logg", "label", "review_status", "notes"]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} targets to {output_file}")


if __name__ == "__main__":
    build_labeling_queue()