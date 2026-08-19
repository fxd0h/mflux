# CI Extract Models

The CI processes need to know _**what models MFlux supports**_ - so we have set up;

**[https://huggingface.co/buckets/mflux-community/ci/tree/models_mflux.json](https://huggingface.co/buckets/mflux-community/ci/tree/models_mflux.json)**

This JSON data file is updated automatically by a GitHub Workflow (`.github/workflows/ci-extract.yml`) which runs when a new main branch is merged to MFlux.

**`just ci-extract`** is a CLI command runner that calls `scripts/ci_extract_models.py` to generate the JSON data file **`.ci_cache/models_mflux.json`**

Developers can manually run the GitHub Action on their own private GitHub account to push the JSON manifest to a dev directory on HuggingFace - eg [https://huggingface.co/buckets/mflux-community/ci/dev/@ianscrivener/tree/models_mflux.json](https://huggingface.co/buckets/mflux-community/ci/dev/@ianscrivener/tree/models_mflux.json).

Developers will need to set up a GitHub Secret `HF_CI_BUCKET_TOKEN` in order to push to HuggingFace.