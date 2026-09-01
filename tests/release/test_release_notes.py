import pytest

from mflux.release.release_notes import ReleaseNotes


def _pr(number, title="a change", body=None, labels=()):
    return {
        "number": number,
        "title": title,
        "body": body,
        "labels": [{"name": name} for name in labels],
    }


@pytest.mark.fast
def test_extracts_the_fenced_block_with_crlf_bodies():
    # GitHub PR bodies arrive with \r\n endings, and review bots append their own
    # sections after the author's text; only the fence is extractable.
    body = (
        "## What\r\nSome reviewer-facing prose.\r\n\r\n"
        "## Release note\r\n```release-note\r\nFixes the thing users saw.\r\n```\r\n"
        "<!-- greptile_comment -->appended bot content<!-- /greptile_comment -->"
    )
    assert ReleaseNotes.extract_release_note(body) == "Fixes the thing users saw."


@pytest.mark.fast
def test_missing_block_and_empty_body_return_none():
    assert ReleaseNotes.extract_release_note("## What\njust prose") is None
    assert ReleaseNotes.extract_release_note(None) is None
    assert ReleaseNotes.extract_release_note("") is None


@pytest.mark.fast
def test_fences_must_be_line_anchored():
    # GitHub only renders fences that start their own line; an inline mention in prose
    # (or a closing backtick glued to other text) is not a block.
    inline = "Add a ```release-note\nlike this``` block to your PR body."
    assert ReleaseNotes.extract_release_note(inline) is None

    unclosed = "```release-note\nA note with no closing fence."
    assert ReleaseNotes.extract_release_note(unclosed) is None

    anchored = "```release-note\nA real note.\n```"
    assert ReleaseNotes.extract_release_note(anchored) == "A real note."


@pytest.mark.fast
def test_render_groups_by_label_and_flags_missing_blocks():
    prs = [
        _pr(10, body="```release-note\nFaster startup.\n```", labels=["improvement"]),
        _pr(11, body="```release-note\nnone\n```", labels=["chore"]),
        _pr(12, title="mystery refactor", body="no block here", labels=["bug"]),
        _pr(13, body="```release-note\nNew sampler.\n```", labels=["enhancement"]),
        _pr(14, body="```release-note\nUnlabeled tweak.\n```"),
    ]
    rendered = ReleaseNotes.render("0.20.0", prs)

    assert "## 0.20.0" in rendered
    assert "### Improved\n- Faster startup. (#10)" in rendered
    assert "### Added\n- New sampler. (#13)" in rendered
    assert "### Fixed\n- mystery refactor (#12) _[needs edit: no release-note block]_" in rendered
    assert "### Changed\n- Unlabeled tweak. (#14)" in rendered
    assert "#11" not in rendered  # explicit `none` is dropped entirely


@pytest.mark.fast
def test_multiline_notes_collapse_to_one_entry_line():
    prs = [_pr(20, body="```release-note\nFirst sentence.\nSecond sentence.\n```", labels=["bug"])]
    rendered = ReleaseNotes.render("0.20.0", prs)
    assert "- First sentence. Second sentence. (#20)" in rendered


@pytest.mark.fast
def test_closing_fence_glued_to_content_is_not_a_block():
    # The closing anchor on its own: a backtick run glued to the note text must not
    # close the block, independent of the opening-fence anchoring covered above.
    assert ReleaseNotes.extract_release_note("```release-note\nA note.```") is None
    assert ReleaseNotes.extract_release_note("```release-note\nA note.\n  ```") is None


@pytest.mark.fast
def test_ci_check_regex_matches_the_extractor():
    # The CI check re-implements _FENCE in .github/workflows/release-note.yml (stdlib
    # python, no repo checkout there). This pins the two byte-identical so they cannot
    # drift — the #685 review round found a looser grep passing incomplete fences.
    import re
    from pathlib import Path

    from mflux.release import release_notes

    workflow = (Path(__file__).parents[2] / ".github" / "workflows" / "release-note.yml").read_text()
    match = re.search(r're\.search\(r"(.+?)", body, (.+?)\)', workflow)
    assert match is not None, "release-note.yml no longer contains the inline regex check"
    assert match.group(1) == release_notes._FENCE.pattern
    for flag in ("re.DOTALL", "re.IGNORECASE", "re.MULTILINE"):
        assert flag in match.group(2)


@pytest.mark.fast
def test_merged_pr_numbers_reads_squash_subjects_oldest_first(tmp_path, monkeypatch):
    import subprocess

    def git(*argv):
        subprocess.run(["git", "-C", str(tmp_path), *argv], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    git("commit", "--allow-empty", "-q", "-m", "base (#1)")
    git("tag", "v.0.1.0")
    git("commit", "--allow-empty", "-q", "-m", "feat: first (#12)")
    git("commit", "--allow-empty", "-q", "-m", "chore: direct push, no PR suffix")
    git("commit", "--allow-empty", "-q", "-m", "fix: second (#13)")

    monkeypatch.chdir(tmp_path)
    assert ReleaseNotes.merged_pr_numbers("v.0.1.0") == [12, 13]


@pytest.mark.fast
def test_reverted_pairs_cancel_within_the_window():
    prs = [
        _pr(10, title="perf: benchmark harness"),
        _pr(11, title='Revert "perf: benchmark harness"'),
        _pr(12, title="fix: unrelated"),
        _pr(13, title='Revert "something merged before this window"'),
    ]
    kept = [pr["number"] for pr in ReleaseNotes.drop_reverted_pairs(prs)]
    # The merge+revert pair is a user-visible no-op; a revert of something outside the
    # window stays, because its target ships in this release as removed.
    assert kept == [12, 13]
