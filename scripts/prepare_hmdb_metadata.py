#!/usr/bin/env python3
"""Convert exported HMDB51 metadata to STOP-compatible JSON/CSV files."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_TEMPLATES = (
    "a person is performing {}.",
    "someone is doing {}.",
    "a video depicts {}.",
)

SPLIT_NAME_MAP = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def label_to_phrase(label: str) -> str:
    text = label.replace("-", " ").replace("_", " ")
    text = re.sub(r"(?<=.)([A-Z])", r" \1", text)
    return " ".join(text.split()).lower()


def derive_video_key(row: pd.Series) -> Tuple[str, str]:
    mp4_path = str(row.get("mp4_path", "")).strip()
    if mp4_path:
        mp4_posix = Path(mp4_path).as_posix()
        video_key = Path(mp4_posix).with_suffix("").as_posix()
    else:
        mp4_posix = ""
        video_key = str(row["video_id"])
        
        
    if '/' in video_key:
        video_key = video_key.split('/')[-1]
    return video_key, mp4_posix


def build_sentences(video_id: str, label: str, next_id: int, templates: List[str]) -> List[Dict[str, object]]:
    phrase = label_to_phrase(label)
    records = []
    for template in templates:
        caption = template.format(phrase)
        records.append({
            "video_id": video_id,
            "caption": caption,
            "sen_id": next_id,
        })
        next_id += 1
    return records


def convert_split(df: pd.DataFrame, split: str, templates: List[str], sentence_start: int):
    videos = []
    sentences = []
    rows = []
    mapped_split = SPLIT_NAME_MAP.get(split, split)
    next_sen_id = sentence_start

    for _, row in df.iterrows():
        original_id = str(row["video_id"])
        label = str(row["label"])
        video_id, mp4_path = derive_video_key(row)
        videos.append({
            "video_id": video_id,
            "original_id": original_id,
            "split": mapped_split,
            "label": label,
            "url": f"hmdb51://{label}/{original_id}",
            "mp4_path": mp4_path,
        })
        new_sentences = build_sentences(video_id, label, next_sen_id, templates)
        sentences.extend(new_sentences)
        next_sen_id = new_sentences[-1]["sen_id"] + 1
        rows.append({
            "video_id": video_id,
            "label": label,
            "original_id": original_id,
            "mp4_path": mp4_path,
        })

    converted_df = pd.DataFrame(rows)
    return videos, sentences, converted_df, next_sen_id


def write_train_csv(df: pd.DataFrame, out_path: Path):
    payload = df[["video_id", "label"]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload.to_csv(out_path, index=False)


def write_eval_csv(df: pd.DataFrame, out_path: Path, split_tag: str, templates: List[str]):
    phrase_template = templates[0]
    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        video_id = str(row["video_id"])
        label = str(row["label"])
        phrase = label_to_phrase(label)
        sentence = phrase_template.format(phrase)
        rows.append({
            "key": f"ret{idx}",
            "vid_key": f"hmdb_{split_tag}_{idx}",
            "video_id": video_id,
            "sentence": sentence,
            "label": label,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="hmdb51_export", help="Directory containing hmdb51_{split}.csv files.")
    parser.add_argument("--output_json", default="hmdb_data.json", help="Output JSON file path.")
    parser.add_argument("--train_csv", default="hmdb_train.csv", help="Output training CSV path.")
    parser.add_argument("--val_csv", default="hmdb_val.csv", help="Output validation CSV path.")
    parser.add_argument("--test_csv", default="hmdb_test.csv", help="Output test CSV path.")
    parser.add_argument("--templates", nargs="*", default=list(DEFAULT_TEMPLATES), help="Caption templates with '{}' placeholder for label phrase.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    splits = ["train", "validation", "test"]
    missing = [split for split in splits if not (source_dir / f"hmdb51_{split}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files in {source_dir}: {missing}")

    templates = args.templates
    all_videos = []
    all_sentences = []
    next_sen_id = 0

    split_frames = {}
    for split in splits:
        csv_path = source_dir / f"hmdb51_{split}.csv"
        df = pd.read_csv(csv_path)
        videos, sentences, converted_df, next_sen_id = convert_split(df, split, templates, next_sen_id)
        all_videos.extend(videos)
        all_sentences.extend(sentences)
        split_frames[split] = converted_df

    payload = {
        "info": {
            "description": "HMDB51 converted for STOP",
            "source": str(source_dir),
            "templates": templates,
        },
        "videos": all_videos,
        "sentences": all_sentences,
    }

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    write_train_csv(split_frames["train"], Path(args.train_csv))
    write_eval_csv(split_frames["validation"], Path(args.val_csv), "val", templates)
    write_eval_csv(split_frames["test"], Path(args.test_csv), "test", templates)


if __name__ == "__main__":
    main()
