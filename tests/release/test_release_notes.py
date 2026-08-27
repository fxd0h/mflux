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
