# NotebookLlama on Union

An open source implementation of NotebookLM that runs on Union. This repo
adapts the [NotebookLlama](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases/NotebookLlama) example in Meta's
[`llama-cookbook`](https://github.com/meta-llama/llama-cookbook) repo.

## Setup

- Create HuggingFace API key
- Get access to the Llama 3.2 model series
- Create Union API key for app serving
- Create Union secret for app serving Union API key

## User Guide

Run the workflow with a local PDF file:

```bash
union run --remote notebook_llama/pdf_to_podcast.py pdf_to_podcast --pdf_path data/2402.13116v4.pdf
```

Run the workflow with a PDF file from a URL:

```bash
union run --remote notebook_llama/pdf_to_podcast.py pdf_to_podcast --pdf_path https://www.biorxiv.org/content/10.1101/544593v2.full.pdf
```

Deploy the streamlit app:

```bash
union deploy apps app.py notebook-lm-test-4
```
