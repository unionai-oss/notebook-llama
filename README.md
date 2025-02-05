# 📖🦙 NotebookLlama on Union

This repo contains an open source implementation of NotebookLM that runs on Union. This repo
adapts the [NotebookLlama](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases/NotebookLlama) example in Meta's
[`llama-cookbook`](https://github.com/meta-llama/llama-cookbook) repo.

## Setup

### Account creation and API key setup

- Create HuggingFace API key [here](https://huggingface.co/settings/tokens).
- Get access to the [Llama 3.1 8B model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and [Llama 3.2 1B model](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
- Create Union Serverless account [here](https://signup.union.ai/).
- Create Union API key for app serving: `union create api-key admin --name notebook-llama`

### Create secrets on Union

Create Union secret for the HuggingFace API key we created in the step above:

```bash
$ union create secret huggingface_api_key
```

You should see a `Enter secret value:` prompt to paste on the secret value.
his will create a secret in Union with the name `huggingface_api_key`.

Now do the same with the `notebook-llama` app serving Union API key:

```bash
$ union create secret union_api_key
Enter secret value:
```

## User Guide

Run the workflow with a PDF file from a URL:

```bash
union run --remote notebook_llama/pdf_to_podcast.py pdf_to_podcast --pdf_path https://www.biorxiv.org/content/10.1101/544593v2.full.pdf
```

Run the workflow with a local PDF file:

```bash
union run --remote notebook_llama/pdf_to_podcast.py pdf_to_podcast --pdf_path data/544593v2.full.pdf
```

Deploy the streamlit app:

```bash
union deploy apps app.py notebook-lm-test-4
```
