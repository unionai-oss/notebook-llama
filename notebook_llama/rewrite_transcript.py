import union

from notebook_llama.common import image


MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """
"""


def create_llm_pipeline():
    ...


@union.task(container_image=image)
def rewrite_transcript():
    ...
