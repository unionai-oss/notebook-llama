# NotebookLlama on Union

An open source implementation of NotebookLM that runs on Union

## User Guide

```bash
union run --remote notebook_llama/pdf_to_podcast.py pdf_to_podcast --pdf_path data/2402.13116v4.pdf
```

### Preprocess PDF

```bash
union run --remote notebook_llama/preprocess_pdf.py pdf_to_text --pdf_path data/2402.13116v4.pdf
```

### Write Transcript

```bash
union run --remote notebook_llama/write_transcript.py write_transcript --pdf_text <uri-to-pdf-text>
```

Where `<uri-to-pdf-text>` is the output of the `pdf_to_text` task.

### Rewrite Transcript

```bash
union run --remote notebook_llama/rewrite_transcript.py rewrite_transcript --transcript <uri-to-transcript>
```

Where `<uri-to-transcript>` is the output of the `write_transcript` task.

### Generate Podcast

```bash
union run --remote notebook_llama/generate_podcast.py generate_podcast --clean_transcript <uri-to-clean-transcript>
```

Where `<uri-to-clean-transcript>` is the output of the `rewrite_transcript` task.
