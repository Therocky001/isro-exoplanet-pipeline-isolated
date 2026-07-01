import csv
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import duckdb


TOI_QUERY_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    "select+tid,tfopwg_disp,st_tmag,pl_pnum,toi+from+toi+where+tfopwg_disp+is+not+null&format=csv"
)

LABEL_POLICY = {
    "planet": {"PC", "KP", "CP", "APC"},
    "eclipsing_binary": {"FP"},
    "blend_noise": {"FA"},
}


def map_label(disposition: str) -> str | None:
    normalized = (disposition or "").strip().upper()
    for label, dispositions in LABEL_POLICY.items():
        if normalized in dispositions:
            return label
    return None


def fetch_toi_rows():
    text = urlopen(TOI_QUERY_URL, timeout=60).read().decode("utf-8", errors="replace")
    reader = csv.DictReader(text.splitlines())
    rows = []
    for row in reader:
        label = map_label(row.get("tfopwg_disp", ""))
        if not label:
            continue
        rows.append(
            {
                "tic_id": f"TIC {row['tid']}",
                "tid": int(row["tid"]),
                "label": label,
                "tfopwg_disp": row.get("tfopwg_disp", ""),
                "st_tmag": float(row["st_tmag"]) if row.get("st_tmag") else None,
                "pl_pnum": int(float(row["pl_pnum"])) if row.get("pl_pnum") else None,
                "toi": row.get("toi", ""),
            }
        )
    return rows


def build_bootstrap_manifest(
    filtered_index_path: str = "output/filtered_index.csv",
    output_path: str = "output/toi_labeled_training_manifest.csv",
    max_per_class: int = 25,
):
    toi_rows = fetch_toi_rows()
    if not toi_rows:
        raise RuntimeError("No TOI rows were retrieved from the public archive.")

    temp_toi_path = Path("output/toi_catalog_bootstrap.csv")
    temp_toi_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_toi_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tid", "tic_id", "label", "tfopwg_disp", "st_tmag", "pl_pnum", "toi"])
        writer.writeheader()
        writer.writerows(toi_rows)

    conn = duckdb.connect(":memory:")
    query = f"""
        WITH toi_labeled AS (
            SELECT * FROM read_csv_auto('{temp_toi_path.as_posix()}')
        )
        SELECT
            f.ID AS tic_id,
            f.rad,
            f.mass,
            f.teff,
            f.tmag,
            f.logg,
            t.label,
            t.tfopwg_disp,
            t.st_tmag,
            t.pl_pnum,
            t.toi
        FROM read_csv_auto('{Path(filtered_index_path).as_posix()}') AS f
        INNER JOIN toi_labeled AS t
            ON CAST(f.ID AS BIGINT) = t.tid
        ORDER BY t.label, t.st_tmag ASC NULLS LAST, f.tmag ASC NULLS LAST
    """

    merged_rows = conn.execute(query).fetchall()
    columns = ["tic_id", "rad", "mass", "teff", "tmag", "logg", "label", "tfopwg_disp", "st_tmag", "pl_pnum", "toi"]

    by_label = defaultdict(list)
    for row in merged_rows:
        by_label[row[6]].append(row)

    selected_rows = []
    for label in ["planet", "eclipsing_binary", "blend_noise"]:
        selected_rows.extend(by_label[label][:max_per_class])

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(selected_rows)

    print(f"Bootstrapped {len(selected_rows)} labeled targets into {output_file}")
    print({label: min(len(rows), max_per_class) for label, rows in by_label.items()})


def build_fast_bootstrap_manifest(
    filtered_index_path: str = "output/filtered_index.csv",
    output_path: str = "output/toi_labeled_training_manifest_fast.csv",
    max_per_class: int = 3,
):
    build_bootstrap_manifest(
        filtered_index_path=filtered_index_path,
        output_path=output_path,
        max_per_class=max_per_class,
    )


if __name__ == "__main__":
    build_bootstrap_manifest()