"""Fetch HYROX PRO MEN results for TrainRox athletes and append to CSV."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

CSV_PATH = Path("data/hyrox_elite15_synthetic_dataset.csv")
INPUT_LIST_PATH = Path("data/athletes_to_fetch.csv")
SPLIT_MAP = {
    "Running 1": "run1",
    "1000m SkiErg": "ski",
    "Running 2": "run2",
    "50m Sled Push": "sled_push",
    "Running 3": "run3",
    "50m Sled Pull": "sled_pull",
    "Running 4": "run4",
    "80m Burpee Broad Jump": "burpees",
    "Running 5": "run5",
    "1000m Row": "row",
    "Running 6": "run6",
    "200m Farmers Carry": "farmers",
    "Running 7": "run7",
    "100m Sandbag Lunges": "lunges",
    "Running 8": "run8",
    "Wall Balls": "wallballs",
}


def fetch_url(url: str) -> Optional[str]:
    try:
        with urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except (URLError, HTTPError) as e:
        print(f"ERROR fetching {url}: {e}")
        return None


def find_pro_men_ids(html: str) -> List[str]:
    """Return list of result IDs for the EXACT division HYROX PRO MEN."""
    ids: List[str] = []
    for line in html.splitlines():
        if "/results/" not in line:
            continue
        
        has_exact = re.search(r"HYROX\s+PRO\s+MEN(?!\S)", line) is not None
        has_doubles = "HYROX PRO DOUBLES MEN" in line
        if not has_exact or has_doubles:
            continue
        for m in re.finditer(r"/results/(\d+)/", line):
            ids.append(m.group(1))
    
    seen = set()
    ordered: List[str] = []
    for i in ids:
        if i not in seen:
            ordered.append(i)
            seen.add(i)
    return ordered


def fetch_splits(result_id: str) -> Optional[List[Dict]]:
    s = fetch_url(f"https://www.trainrox.com/api/split-distributions/{result_id}")
    if s is None:
        return None
    
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"ERROR decoding JSON for {result_id}: {e}")
        return None


def to_csv_row(athlete_name: str, result_id: str, splits: List[Dict]) -> Dict[str, Optional[float]]:
    """Convert split JSON to a CSV row dict matching the dataset schema."""
    row: Dict[str, Optional[float]] = {
        "athlete": athlete_name,
        "gender": "M",
        "race": f"Event {result_id}",
        "source": "trainrox",
    }

    for col in [
        "run1","ski","run2","sled_push","run3","sled_pull","run4","burpees",
        "run5","row","run6","farmers","run7","lunges","run8","wallballs",
    ]:
        row[col] = None

    for entry in splits:
        name = entry.get("splitName")
        athlete = entry.get("athlete") or {}
        time = athlete.get("time")
        if name in SPLIT_MAP and time is not None:
            row[SPLIT_MAP[name]] = time

    run_cols = ["run1","run2","run3","run4","run5","run6","run7","run8"]
    work_cols = ["ski","sled_push","sled_pull","burpees","row","farmers","lunges","wallballs"]

    run_total = sum((row[c] or 0) for c in run_cols)
    working_total = sum((row[c] or 0) for c in work_cols)
    total_time = run_total + working_total

    row["run_total"] = float(run_total)
    row["working_total"] = float(working_total)
    row["total_time"] = int(total_time)

    return row


def append_rows(csv_path: Path, rows: List[Dict[str, Optional[float]]]) -> None:
    df = pd.read_csv(csv_path)
    
    existing = set(df["race"].astype(str).tolist())
    add_rows = [r for r in rows if str(r.get("race")) not in existing]
    if not add_rows:
        print("No new rows to append.")
        return

    for r in add_rows:
        for col in df.columns:
            r.setdefault(col, None)

    new_df = pd.concat([df, pd.DataFrame(add_rows)], ignore_index=True)
    new_df.to_csv(csv_path, index=False)
    print(f"Appended {len(add_rows)} rows to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(INPUT_LIST_PATH),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input list not found: {input_path}")
        return

    athletes_df = pd.read_csv(input_path)
    if not set(["url", "athlete_name"]).issubset(athletes_df.columns):
        print("Input CSV must contain columns: url, athlete_name")
        return

    existing_races: Set[str] = set()
    try:
        existing_df = pd.read_csv(CSV_PATH)
        if "race" in existing_df.columns:
            existing_races = set(existing_df["race"].astype(str).tolist())
    except Exception:
        existing_races = set()

    for _, row in athletes_df.iterrows():
        athlete_url = str(row["url"]).strip()
        athlete_name = str(row["athlete_name"]).strip()
        if not athlete_url:
            continue
        html = fetch_url(athlete_url)
        if html is None:
            continue
        ids = find_pro_men_ids(html)
        if not ids:
            print(f"No HYROX PRO MEN IDs for {athlete_name} ({athlete_url})")
            continue
        print(f"{athlete_name}: found {len(ids)} IDs -> {ids}")
        athlete_rows: List[Dict[str, Optional[float]]] = []
        for rid in ids:
            race_id = f"Event {rid}"
            if race_id in existing_races:
                continue
            splits = fetch_splits(rid)
            if splits is None:
                continue
            athlete_rows.append(to_csv_row(athlete_name, rid, splits))

        if athlete_rows:
            append_rows(CSV_PATH, athlete_rows)
        else:
            print(f"No new rows to append for {athlete_name}.")


if __name__ == "__main__":
    main()
