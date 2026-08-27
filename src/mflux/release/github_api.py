import requests


class GitHubAPI:
    @staticmethod
    def check_github_release_exists(
        github_token: str,
        github_repo: str,
        tag_name: str,
    ) -> bool:
        print("🔍 Checking GitHub release existence...")

        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MFLUX-Release-Script",
        }

        url = f"https://api.github.com/repos/{github_repo}/releases/tags/{tag_name}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            print(f"✅ GitHub release {tag_name} exists")
            return True

        if response.status_code == 404:
            print(f"   GitHub release {tag_name} does not exist")
            return False

        # Any other status code is unexpected and likely indicates auth/rate-limit issues.
        error_msg = f"GitHub API returned {response.status_code} while checking for release {tag_name}: {response.text}"
        raise requests.HTTPError(error_msg, response=response)

    @staticmethod
    def create_github_release(
        github_token: str,
        github_repo: str,
        tag_name: str,
        version: str,
        release_notes: str,
    ) -> dict:
        print("🐙 Creating GitHub release...")

        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MFLUX-Release-Script",
        }

        url = f"https://api.github.com/repos/{github_repo}/releases"

        data = {
            "tag_name": tag_name,
            "name": f"Release {version}",
            "body": release_notes,
            "draft": False,
            "prerelease": False,
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 201:
            print(f"✅ Successfully created GitHub release for {tag_name}")
            return response.json()
        else:
            raise Exception(f"Failed to create GitHub release: {response.status_code} - {response.text}")

    @staticmethod
    def find_release(github_token: str, github_repo: str, tag_name: str) -> dict | None:
        # Drafts have no tag yet, so /releases/tags/ cannot see them; list and match instead.
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MFLUX-Release-Script",
        }
        url = f"https://api.github.com/repos/{github_repo}/releases"
        response = requests.get(url, headers=headers, params={"per_page": 100}, timeout=(5, 30))
        if response.status_code != 200:
            raise requests.HTTPError(
                f"GitHub API returned {response.status_code} listing releases: {response.text}", response=response
            )
        for release in response.json():
            if release.get("tag_name") == tag_name:
                return release
        return None

    @staticmethod
    def upsert_draft_release(
        github_token: str,
        github_repo: str,
        tag_name: str,
        version: str,
        release_notes: str,
    ) -> dict:
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MFLUX-Release-Script",
        }
        existing = GitHubAPI.find_release(github_token, github_repo, tag_name)
        if existing is not None and not existing.get("draft", False):
            raise ValueError(f"Release {tag_name} is already published; refusing to overwrite its notes")
        data = {
            "tag_name": tag_name,
            "name": f"Release {version}",
            "body": release_notes,
            "draft": True,
            "prerelease": False,
        }
        if existing is None:
            response = requests.post(f"https://api.github.com/repos/{github_repo}/releases", json=data, headers=headers, timeout=(5, 30))  # fmt: off
            expected = 201
        else:
            response = requests.patch(existing["url"], json=data, headers=headers, timeout=(5, 30))
            expected = 200
        if response.status_code != expected:
            raise Exception(f"Failed to upsert draft release: {response.status_code} - {response.text}")
        print(f"\U0001f4dd Draft release ready for {tag_name}: {response.json().get('html_url')}")
        return response.json()

    @staticmethod
    def publish_draft_release(github_token: str, github_repo: str, release: dict) -> dict:
        # Publishing flips draft off and lets GitHub create the tag; the body ships exactly
        # as the approver left it in the draft, edits included.
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MFLUX-Release-Script",
        }
        response = requests.patch(release["url"], json={"draft": False}, headers=headers, timeout=(5, 30))
        if response.status_code != 200:
            raise Exception(f"Failed to publish draft release: {response.status_code} - {response.text}")
        print(f"\u2705 Published GitHub release {release.get('tag_name')}")
        return response.json()
