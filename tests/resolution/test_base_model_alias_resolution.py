import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.utils.exceptions import ModelConfigError


@pytest.mark.fast
def test_base_model_keyword_alone_resolves_a_real_model_name():
    # `--base-model schnell` with no --model asks for the base model itself. A test over
    # the static table stays green through this bug: the table is right and from_name is
    # what was wrong, handing back model_name=None when only the base_model keyword named
    # the model, which every initializer then fails to resolve a weights path from.
    for config in AVAILABLE_MODELS.values():
        if config.base_model is not None:
            continue
        for alias in config.aliases:
            resolved = ModelConfig.from_name(model_name=None, base_model=alias)
            assert resolved.model_name == config.model_name, (
                f"base_model={alias!r} resolved model_name={resolved.model_name!r}, expected {config.model_name!r}"
            )


@pytest.mark.fast
def test_both_keywords_yield_the_same_shape():
    # Whichever keyword names the model, the resolved config must look the same.
    by_model = ModelConfig.from_name(model_name="schnell", base_model=None)
    by_base = ModelConfig.from_name(model_name=None, base_model="schnell")
    assert by_base.model_name == by_model.model_name
    assert by_base.supports_guidance == by_model.supports_guidance


@pytest.mark.fast
def test_naming_neither_keyword_raises_a_model_error():
    # Both keywords optional means both can be absent, and that has to arrive as a
    # ModelConfigError rather than as an AttributeError from the substring rule
    # lowercasing None two frames deeper.
    with pytest.raises(ModelConfigError):
        ModelConfig.from_name(model_name=None)


@pytest.mark.fast
def test_a_non_flux_root_is_a_valid_base_model():
    # base_model accepts any root alias, which is what the resolver validates against.
    resolved = ModelConfig.from_name(model_name="some-user/some-finetune", base_model="z-image")

    assert resolved.model_name == "some-user/some-finetune"
    assert resolved.base_model == AVAILABLE_MODELS["z-image"].model_name


@pytest.mark.fast
def test_custom_checkpoint_with_base_model_keeps_both_fields():
    # The two-keyword form (a third-party checkpoint on a known architecture) is the
    # case the explicit-base rule exists for; it must keep working unchanged.
    resolved = ModelConfig.from_name(model_name="some-user/some-finetune", base_model="schnell")
    assert resolved.model_name == "some-user/some-finetune"
    assert resolved.base_model == "black-forest-labs/FLUX.1-schnell"


@pytest.mark.fast
@pytest.mark.parametrize(
    ("repo", "key"),
    [
        ("mflux-community/flux-3-image-klein-mflux-bf16", "flux3-image-klein"),
        ("mflux-community/flux-3-image-klein-mflux-q8", "flux3-image-klein"),
        ("mflux-community/flux-3-image-klein-mflux-q4", "flux3-image-klein"),
        ("mflux-community/flux-3-image-klein-base-mflux-bf16", "flux3-image-klein-base"),
        ("mflux-community/flux-3-image-klein-base-mflux-q8", "flux3-image-klein-base"),
        ("mflux-community/flux-3-image-klein-base-mflux-q4", "flux3-image-klein-base"),
    ],
)
def test_published_flux3_community_repos_resolve_by_name(repo, key):
    # #908: no flux3 alias was a substring of these repo names, so from_name raised
    # "Cannot infer base_model". The -base repos contain both aliases. The longest alias wins.
    assert ConfigResolution.resolve_key(repo) == key
    resolved = ModelConfig.from_name(model_name=repo)
    assert resolved.model_name == repo
    assert resolved.base_model == AVAILABLE_MODELS[key].model_name


@pytest.mark.fast
def test_flux3_family_inference_selects_base_from_repo_name():
    # Without --base-model, the flux3 CLI infers the variant from the repo name.
    assert (
        ConfigResolution.infer_family_member(
            "mflux-community/flux-3-image-klein-base-mflux-q8", "flux3-image-klein", ("flux3-image-klein-base",)
        )
        is AVAILABLE_MODELS["flux3-image-klein-base"]
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    ("name", "key"),
    [("flux-3-image-klein", "flux3-image-klein"), ("flux-3-image-klein-base", "flux3-image-klein-base")],
)
def test_hyphenated_flux3_aliases_are_exact_names(name, key):
    # The new aliases are also valid --model and --base-model values.
    assert ModelConfig.from_name(model_name=name) is AVAILABLE_MODELS[key]
    assert name in ConfigResolution.base_model_names()


@pytest.mark.fast
@pytest.mark.parametrize(
    ("repo", "default", "family", "key"),
    [
        ("mflux-community/fibo-2-0-dev-mflux-q4", "fibo2", ("fibo2-edit-turbo",), "fibo2"),
        ("mflux-community/fibo-2-0-edit-turbo-mflux-q8", "fibo2", ("fibo2-edit-turbo",), "fibo2-edit-turbo"),
        ("mflux-community/fibo-2-0-edit-turbo-mflux-bf16", "fibo2", ("fibo2-edit-turbo",), "fibo2-edit-turbo"),
        ("mflux-community/qwen-image-2-1-mflux-q8", "qwen-image-2.1", ("qwen-image-2.1-turbo",), "qwen-image-2.1"),
        (
            "mflux-community/qwen-image-2-1-turbo-mflux-q4",
            "qwen-image-2.1",
            ("qwen-image-2.1-turbo",),
            "qwen-image-2.1-turbo",
        ),  # fmt: off
    ],
)
def test_published_fibo2_and_qwen21_community_repos_pick_their_variant_by_name(repo, default, family, key):
    # The repo names write the version with hyphens (2-0, 2-1), which no alias matched, so the Edit Turbo and Turbo
    # repos ran as Dev and as the base model unless --base-model named them.
    assert ConfigResolution.infer_family_member(repo, default, family) is AVAILABLE_MODELS[key]


@pytest.mark.fast
def test_hyphenated_qwen21_controlnet_union_alias_is_an_exact_name():
    # The hyphenated spelling is a valid --model and --base-model value, like the other 2-1 aliases.
    name = "qwen-image-2-1-controlnet-union"
    assert ModelConfig.from_name(model_name=name) is AVAILABLE_MODELS["qwen-image-2.1-controlnet-union"]
    assert name in ConfigResolution.base_model_names()
