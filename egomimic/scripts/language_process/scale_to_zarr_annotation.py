"""
Example usage:
python egomimic/scripts/language_process/scale_to_zarr_annotation.py \
--scale-annotation-dir annotations_test \
--dataset-config-path egomimic/hydra_configs/data/scale.yaml \
--conversion-mode pick_place_llm \
--prompt-filepath egomimic/scripts/language_process/prompt.txt \
--project-name "dense-language"
"""

import argparse
import json
import os
from subprocess import run
from typing import Any

import hydra
import pandas as pd
import requests
from omegaconf import OmegaConf
from scaleapi import ScaleClient

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.language_process.converter import (
    HardCodedConverter,
    PickPlaceLLMConverter,
)

REQUEST_TIMEOUT_S = 60


def get_completed_tasks(project_name: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch all completed tasks for a project."""
    headers = {"accept": "application/json"}
    base_url = "https://api.scale.com/v1/tasks"

    next_token = None
    tasks: list[dict[str, Any]] = []

    while True:
        params = {  # TODO: fetch all tasks included non completed tasks after scale explained how this is done
            "project": project_name,
            "include_attachment_url": "true",
            "limit": 100,
        }
        if next_token:
            params["next_token"] = next_token

        response = requests.get(
            base_url,
            headers=headers,
            params=params,
            auth=(api_key, ""),
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()

        tasks.extend(data.get("docs", []))

        next_token = data.get("next_token")
        if not next_token:
            break

    return tasks


def get_episode_hash(task: dict[str, Any]) -> str:
    """Extract episode hash from task['params']['attachments'][0]."""
    attachment = task["params"]["attachments"][0]
    # Example:
    # s3://scale-sales-uploads/egoverse/2026-03-17-01-42-37-000000/2026-03-17-01-42-37-000000.mp4
    return attachment.rstrip("/").split("/")[-2]


def build_df_from_tasks(
    tasks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a lookup from episode_hash -> task."""

    df = pd.DataFrame(columns=["_ID", "STATUS", "S3_ATTACHMENT", "SEQUENCE_ID"])
    for task in tasks:
        attachments = task.get("params", {}).get("attachments", [])
        if not attachments:
            continue

        episode_hash = get_episode_hash(task)
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "_ID": task["task_id"],
                            "STATUS": task["status"],
                            "S3_ATTACHMENT": attachments[0],
                            "SEQUENCE_ID": episode_hash,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return df


def download_scale_annotation_csv(dest_path: str):
    r2_path = "s3://rldb/scale_annotation_csv/Dense_Language_Tasks_2026_03_31.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pull_csv_script = os.path.join(script_dir, "pull_csv.sh")
    run(["bash", pull_csv_script, r2_path, dest_path], check=True)
    return os.path.join(dest_path, "Dense_Language_Tasks_2026_03_31.csv")


def load_scale_annotation_csv(csv_path: str):
    return pd.read_csv(csv_path)


def get_available_hashes(df: pd.DataFrame):
    df = df[df["STATUS"] == "completed"]
    return df["SEQUENCE_ID"].unique().tolist()


def download_scale_annotation(client: ScaleClient, tid: str, out_path: str):
    task = client.get_task(tid)
    url = task.response["annotations"]["url"]
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    raw = json.loads(resp.text.rstrip("\x00"))
    path = os.path.join(out_path, f"{tid}.json")
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)


def get_tid_to_episode_hash(df: pd.DataFrame, tid: str):
    return df[df["_ID"] == tid]["SEQUENCE_ID"].values[0]


def get_episode_hash_to_tid(df: pd.DataFrame, episode_hash: str):
    return df[df["SEQUENCE_ID"] == episode_hash]["_ID"].values[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", type=str, default=None)
    parser.add_argument("--scale-annotation-dir", type=str, required=True)
    parser.add_argument("--dataset-config-path", type=str, required=True)
    parser.add_argument(
        "--conversion-mode",
        type=str,
        required=True,
        choices=["pick_place_llm", "hardcoded"],
    )
    parser.add_argument(
        "-s", "--scale-api-key", default=os.environ.get("SCALE_API_KEY", "")
    )
    parser.add_argument("--prompt-filepath", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.scale_annotation_dir, exist_ok=True)

    if args.project_name:
        completed_tasks = get_completed_tasks(args.project_name, args.scale_api_key)
        df = build_df_from_tasks(completed_tasks)
    else:
        csv_path = download_scale_annotation_csv(args.scale_annotation_dir)
        df = load_scale_annotation_csv(csv_path)
    dataset_cfg = OmegaConf.load(args.dataset_config_path)
    train_datasets = {}
    train_hashes = set()
    for dataset_name in dataset_cfg.train_datasets:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            dataset_cfg.train_datasets[dataset_name]
        )  # MultiDataset
        train_hashes.update(list(train_datasets[dataset_name].datasets.keys()))

    valid_datasets = {}
    valid_hashes = set()
    for dataset_name in dataset_cfg.valid_datasets:
        valid_datasets[dataset_name] = hydra.utils.instantiate(
            dataset_cfg.valid_datasets[dataset_name]
        )  # MultiDataset
        valid_hashes.update(list(valid_datasets[dataset_name].datasets.keys()))

    dataset_hashes = train_hashes.union(valid_hashes)
    available_hashes = get_available_hashes(df)

    # Check if all annotations are present for the train and valid datasets and download scale annotations
    missing_hashes = dataset_hashes - set(available_hashes)
    if missing_hashes:
        error_message = "Missing annotations for the following hashes: "
        for hash in missing_hashes:
            error_message += f"\n{hash}"
        raise ValueError(error_message)

    client = ScaleClient(args.scale_api_key)
    # download the annotations for the dataset hashes
    for hash in dataset_hashes:
        tid = df[df["SEQUENCE_ID"] == hash]["_ID"].values[0]
        download_scale_annotation(client, tid, args.scale_annotation_dir)

    # instantiate converter
    if args.conversion_mode == "pick_place_llm":
        converter = PickPlaceLLMConverter(
            args.scale_annotation_dir, args.prompt_filepath
        )
    elif args.conversion_mode == "hardcoded":
        converter = HardCodedConverter(args.scale_annotation_dir)
    else:
        raise ValueError(f"Invalid conversion mode: {args.conversion_mode}")

    # Write annotations to train datasets
    for dataset_name in train_datasets:
        for episode_hash, zarr_dataset in train_datasets[dataset_name].datasets.items():
            tid = get_episode_hash_to_tid(df, episode_hash)
            annotation = converter.convert(tid)
            writer = ZarrWriter(
                episode_path=zarr_dataset.episode_path,
                verbose=True,
            )
            writer.append_annotations(
                annotation_key="annotations", annotations=annotation, mode="w"
            )

    for dataset_name in valid_datasets:
        for episode_hash, zarr_dataset in valid_datasets[dataset_name].datasets.items():
            tid = get_episode_hash_to_tid(df, episode_hash)
            annotation = converter.convert(tid)
            writer = ZarrWriter(
                episode_path=zarr_dataset.episode_path,
                verbose=True,
            )
            writer.append_annotations(
                annotation_key="annotations", annotations=annotation, mode="w"
            )
