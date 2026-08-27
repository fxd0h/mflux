import re

import requests

# GitHub PR bodies arrive with \r\n line endings; tolerate the \r on the fence line.
_FENCE = re.compile(r"```release-note[ \t\r]*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Label -> section, first match wins in this order. PRs with no matching label land in
# "Changed"; PRs with no release-note block land in their section flagged for the releaser.
_SECTIONS: list[tuple[str, str]] = [
    ("breaking change", "Breaking"),
    ("bug", "Fixed"),
    ("feature suggestion", "Added"),
    ("enhancement", "Added"),
    ("improvement", "Improved"),
    ("documentation", "Docs"),
    ("chore", "Internal"),
    ("ci", "Internal"),
]
_SECTION_ORDER = ["Breaking", "Added", "Improved", "Fixed", "Docs", "Internal", "Changed"]


class ReleaseNotes:
    @staticmethod
    def extract_release_note(body: str | None) -> str | None:
        # Only the fenced block is trusted: review bots append their own sections to PR
        # bodies, so anything outside the fence is not extractable without guessing.
        if not body:
            return None
        match = _FENCE.search(body)
        if match is None:
            return None
        return match.group(1).strip()

    @staticmethod
    def collect_merged_prs(github_token: str, github_repo: str, since: str) -> list[dict]:
        headers = ReleaseNotes._headers(github_token)
        prs: list[dict] = []
        page = 1
        while True:
            response = requests.get(
                "https://api.github.com/search/issues",
                headers=headers,
                params={
                    "q": f"repo:{github_repo} is:pr is:merged merged:>{since}",
                    "per_page": 100,
                    "page": page,
                },
                timeout=(5, 30),
            )
            if response.status_code != 200:
                raise requests.HTTPError(
                    f"GitHub search returned {response.status_code} collecting merged PRs: {response.text}",
                    response=response,
                )
            items = response.json().get("items", [])
            prs.extend(items)
            if len(items) < 100:
                return prs
            page += 1

    @staticmethod
    def latest_release_date(github_token: str, github_repo: str) -> str:
        response = requests.get(
            f"https://api.github.com/repos/{github_repo}/releases/latest",
            headers=ReleaseNotes._headers(github_token),
            timeout=(5, 30),
        )
        if response.status_code != 200:
            raise requests.HTTPError(
                f"GitHub API returned {response.status_code} resolving the latest release: {response.text}",
                response=response,
            )
        return response.json()["published_at"]

    @staticmethod
    def render(version: str, prs: list[dict]) -> str:
        sections: dict[str, list[str]] = {name: [] for name in _SECTION_ORDER}
        for pr in sorted(prs, key=lambda p: p["number"]):
            note = ReleaseNotes.extract_release_note(pr.get("body"))
            if note is not None and note.lower() == "none":
                continue
            section = ReleaseNotes._section_for(pr)
            if note:
                lines = [line.strip() for line in note.splitlines() if line.strip()]
                sections[section].append(f"- {' '.join(lines)} (#{pr['number']})")
            else:
                sections[section].append(f"- {pr['title'].strip()} (#{pr['number']}) _[needs edit: no release-note block]_")  # fmt: off

        rendered = [f"## {version}", ""]
        for name in _SECTION_ORDER:
            if not sections[name]:
                continue
            rendered.append(f"### {name}")
            rendered.extend(sections[name])
            rendered.append("")
        return "\n".join(rendered).rstrip() + "\n"

    @staticmethod
    def _section_for(pr: dict) -> str:
        labels = {label["name"].lower() for label in pr.get("labels", [])}
        for label, section in _SECTIONS:
            if label in labels:
                return section
        return "Changed"

    @staticmethod
    def _headers(github_token: str) -> dict:
        return {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MFLUX-Release-Script",
        }
