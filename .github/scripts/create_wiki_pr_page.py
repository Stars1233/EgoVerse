#!/usr/bin/env python3
"""
Creates or updates wiki/prs/pr-NNNN.md in the Obsidian vault repo
whenever a PR is opened, closed, or merged.

Copy to: EgoVerse/.github/scripts/create_wiki_pr_page.py
"""

import base64
import os
from datetime import datetime

import requests


def get_file_sha(url: str, headers: dict) -> str | None:
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json()["sha"]
    return None


def build_page(pr_number: int, title: str, author: str, status: str,
               created_at: str, body: str) -> str:
    updated = datetime.utcnow().strftime("%Y-%m-%d")
    created = created_at[:10]
    github_url = f"https://github.com/GaTech-RL2/EgoVerse/pull/{pr_number}"
    desc = (body or "(no description)").strip()
    source_line = (
        f"- [[raw/prs/pr-{pr_number:04d}.json]]"
        if status == "merged"
        else ""
    )

    checklist = """- [ ] Zarr writes use `ZarrWriter` (never manual)
- [ ] Embodiment strings match exact enum
- [ ] SQL `operator` field is SHA-256 hashed
- [ ] Poses in SLAM world frame (re-expressed at training time only)
- [ ] Upload targets Cloudflare R2, not AWS S3
- [ ] No absolute paths
- [ ] `dataset_name == embodiment` assumption not made worse"""

    return f"""# PR #{pr_number} — {title}

**Type**: pr
**Author**: {author}
**Status**: {status}
**GitHub**: {github_url}
**Implements**:
**Created**: {created}
**Updated**: {updated}

{desc[:300]}{'...' if len(desc) > 300 else ''}

---

## Changes

{desc}

## Review Notes

## EgoVerse Checklist

{checklist}

## Contradictions & Open Questions

## See Also

## Sources
{source_line}
"""


def main():
    pr_number = int(os.environ["PR_NUMBER"])
    title = os.environ.get("PR_TITLE", "")
    author = os.environ.get("PR_AUTHOR", "")
    status = os.environ.get("PR_STATUS", "open")
    created_at = os.environ.get("PR_CREATED_AT", datetime.utcnow().isoformat())
    body = os.environ.get("PR_BODY", "") or ""
    vault_repo = os.environ["OBSIDIAN_VAULT_REPO"]
    pat = os.environ["VAULT_PAT"]

    file_path = f"wiki/prs/pr-{pr_number:04d}.md"
    url = f"https://api.github.com/repos/{vault_repo}/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    sha = get_file_sha(url, headers)
    content = build_page(pr_number, title, author, status, created_at, body)
    encoded = base64.b64encode(content.encode()).decode()

    payload = {
        "message": f"wiki: track PR #{pr_number} [{status}] — {title}",
        "content": encoded,
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha

    resp = requests.put(url, headers=headers, json=payload)
    resp.raise_for_status()
    action = "Updated" if sha else "Created"
    print(f"{action}: {file_path}")


if __name__ == "__main__":
    main()
