import union

from notebook_llama.common import image


def create_tts_pipeline_speaker1():
    ...


def create_tts_pipeline_speaker2():
    ...


@union.task(container_image=image)
def generate_podcast():
    ...
