# The GitHub-side release logic: find/create/publish of the draft and the draft_notes
# no-overwrite semantics. All network calls are faked at the requests boundary.

import pytest

from mflux.release import (
    github_api as github_api_module,
    release_manager as release_manager_module,
)
from mflux.release.github_api import GitHubAPI
from mflux.release.release_manager import ReleaseManager


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.mark.fast
def test_find_release_paginates_and_sees_drafts(monkeypatch):
    pages = {
        1: [{"tag_name": f"v.0.0.{i}", "draft": False} for i in range(100)],
        2: [{"tag_name": "v.0.20.0", "draft": True, "url": "u"}],
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        return _Response(200, pages[params["page"]])

    monkeypatch.setattr(github_api_module.requests, "get", fake_get)
    release = GitHubAPI.find_release("t", "o/r", "v.0.20.0")
    assert release is not None and release["draft"] is True
    assert GitHubAPI.find_release("t", "o/r", "v.9.9.9") is None


@pytest.mark.fast
def test_create_draft_release_posts_a_draft(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(json)
        return _Response(201, {"html_url": "h", **json})

    monkeypatch.setattr(github_api_module.requests, "post", fake_post)
    GitHubAPI.create_draft_release("t", "o/r", "v.0.20.0", "0.20.0", "## notes")
    assert captured["draft"] is True
    assert captured["tag_name"] == "v.0.20.0"
    assert captured["body"] == "## notes"


@pytest.mark.fast
def test_publish_draft_release_flips_draft_only(monkeypatch):
    captured = {}

    def fake_patch(url, json=None, headers=None, timeout=None):
        captured.update(json)
        return _Response(200, {"tag_name": "v.0.20.0"})

    monkeypatch.setattr(github_api_module.requests, "patch", fake_patch)
    GitHubAPI.publish_draft_release("t", "o/r", {"url": "u", "tag_name": "v.0.20.0"})
    # The body is not sent: the release ships exactly as the approver edited it.
    assert captured == {"draft": False}


@pytest.mark.fast
@pytest.mark.parametrize(
    "existing,message",
    [
        ({"draft": True}, "leaving the approver's copy untouched"),
        ({"draft": False}, "already published; nothing to draft"),
    ],
    ids=["draft-exists", "published"],
)
def test_draft_notes_never_overwrites_an_existing_release(monkeypatch, capsys, existing, message):
    # A re-dispatch must not clobber the approver's edited draft, and re-dispatching an
    # already-released version is the documented no-op, not an error.
    monkeypatch.setattr(GitHubAPI, "find_release", staticmethod(lambda *a: existing))
    monkeypatch.setattr(
        GitHubAPI,
        "create_draft_release",
        staticmethod(lambda *a: pytest.fail("must not create over an existing release")),
    )
    monkeypatch.setattr(
        ReleaseManager,
        "_harvest_notes",
        staticmethod(lambda *a: pytest.fail("must not harvest when nothing will be written")),
    )
    ReleaseManager.draft_notes("t", "o/r")
    assert message in capsys.readouterr().out


@pytest.mark.fast
def test_draft_notes_creates_when_absent(monkeypatch):
    created = {}
    monkeypatch.setattr(GitHubAPI, "find_release", staticmethod(lambda *a: None))
    monkeypatch.setattr(
        GitHubAPI,
        "create_draft_release",
        staticmethod(lambda token, repo, tag, version, notes: created.update(tag=tag, notes=notes)),
    )
    monkeypatch.setattr(ReleaseManager, "_harvest_notes", staticmethod(lambda *a: "## harvested"))
    monkeypatch.setattr(release_manager_module.VersionUtil, "get_mflux_version", staticmethod(lambda: "0.20.0"))
    ReleaseManager.draft_notes("t", "o/r")
    assert created == {"tag": "v.0.20.0", "notes": "## harvested"}
