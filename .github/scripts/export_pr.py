#!/usr/bin/env python3
"""
Exports a merged PR's diff, review comments, and metadata to pr_export.json.
Copy to: EgoVerse/.github/scripts/export_pr.py
"""

import json
import os
import subprocess
from datetime import datetime

import requests


def get_pr_comments(repo: str, pr_number: int, token: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return [
        {
            "author": c["user"]["login"],
            "body": c["body"],
            "path": c["path"],
            "line": c.get("line"),
            "created_at": c["created_at"],
        }
        for c in resp.json()
    ]


def get_pr_reviews(repo: str, pr_number: int, token: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return [
        {
            "author": r["user"]["login"],
            "state": r["state"],
            "body": r["body"],
            "submitted_at": r["submitted_at"],
        }
        for r in resp.json()
    ]


def main():
    pr_number = int(os.environ["PR_NUMBER"])
    base_sha = os.environ["BASE_SHA"]
    head_sha = os.environ["HEAD_SHA"]
    token = os.environ["GH_TOKEN"]
    repo = os.environ.get("GITHUB_REPOSITORY", "GaTech-RL2/EgoVerse")

    # Get diff
    diff = subprocess.check_output(
        ["git", "diff", f"{base_sha}...{head_sha}"],
        text=True,
    )

    # Truncate diff if huge
    if len(diff) > 200_000:
        diff = diff[:200_000] + "\n... [truncated]"

    export = {
        "pr_number": pr_number,
        "title": os.environ.get("PR_TITLE", ""),
        "body": os.environ.get("PR_BODY", ""),
        "author": os.environ.get("PR_AUTHOR", ""),
        "merged_at": os.environ.get("PR_MERGED_AT", ""),
        "base_ref": os.environ.get("BASE_REF", "main"),
        "head_sha": head_sha,
        "base_sha": base_sha,
        "repo": repo,
        "diff": diff,
        "review_comments": get_pr_comments(repo, pr_number, token),
        "reviews": get_pr_reviews(repo, pr_number, token),
        "exported_at": datetime.utcnow().isoformat(),
    }

    filename = f"pr_{pr_number:04d}_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open("pr_export.json", "w") as f:
        json.dump(export, f, indent=2)

    print(f"Exported PR #{pr_number} to pr_export.json ({len(diff)} chars of diff)")


if __name__ == "__main__":
    main()
