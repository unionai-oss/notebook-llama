import union

from notebook_llama.common import image


DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def extract_text_from_pdf():
    ...


def create_word_bounded_chunks():
    ...


def process_chunk():
    ...


@union.task(container_image=image)
def preprocess_pdf():
    ...
