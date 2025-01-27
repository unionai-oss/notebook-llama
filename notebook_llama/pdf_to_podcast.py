import union

from notebook_llama.preprocess_pdf import pdf_to_text
from notebook_llama.write_transcript import write_transcript
from notebook_llama.rewrite_transcript import rewrite_transcript
from notebook_llama.generate_podcast import generate_podcast, create_podcast_deck


@union.workflow
def pdf_to_podcast(pdf_path: union.FlyteFile) -> union.FlyteFile:
    pdf_text = pdf_to_text(pdf_path)
    transcript = write_transcript(pdf_text)
    clean_transcript = rewrite_transcript(transcript)
    podcast = generate_podcast(clean_transcript)
    create_podcast_deck(podcast, clean_transcript)
    return podcast
