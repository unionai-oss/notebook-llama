import io
import json
from pathlib import Path

import union
from flytekit.extras import accelerators
from flytekit.deck import MarkdownRenderer

from notebook_llama.images import audio_image

from tqdm import tqdm

import torch
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

speaker1_description = """
Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise.
"""

speaker2_description = """
Gary's voice is calm and smooth in delivery, speaking at a moderate pace with a very close recording that almost has no background noise.
"""

TTS_MODEL = "parler-tts/parler-tts-mini-v1"


def create_tts_pipeline(use_4bit: bool = False):
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from parler_tts import ParlerTTSForConditionalGeneration

    bitsandbytes_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    kwargs = {}
    if use_4bit:
        kwargs["quantization_config"] = bitsandbytes_config
        kwargs["torch_dtype"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
        TTS_MODEL, **kwargs,
    ).to(device)
    parler_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)
    return parler_model, parler_tokenizer


def generate_speaker_audio(parler_model, parler_tokenizer, text, speaker_description):
    """Generate audio using ParlerTTS for Speaker 1"""
    import torch

    with torch.no_grad():
        input_ids = parler_tokenizer(
            speaker_description, return_tensors="pt"
        ).input_ids.to(device)
        prompt_input_ids = parler_tokenizer(text, return_tensors="pt").input_ids.to(device)
        generation = parler_model.generate(
            input_ids=input_ids, prompt_input_ids=prompt_input_ids
        )
        audio_arr = generation.cpu().to(torch.float32).numpy().squeeze()
    return audio_arr, parler_model.config.sampling_rate


def numpy_to_audio_segment(audio_arr, sampling_rate):
    """Convert numpy array to AudioSegment"""
    from scipy.io import wavfile
    from pydub import AudioSegment

    # Convert to 16-bit PCM
    audio_int16 = (audio_arr * 32767).astype(np.int16)

    # Create WAV file in memory
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sampling_rate, audio_int16)
    byte_io.seek(0)

    # Convert to AudioSegment
    return AudioSegment.from_wav(byte_io)


def produce_final_audio(podcast_text: list[list[str]]) -> Path:
    final_audio = None

    parler_model, parler_tokenizer = create_tts_pipeline()

    for speaker, text in tqdm(
        podcast_text, desc="Generating podcast segments", unit="segment"
    ):
        if speaker == "Laura":
            audio_arr, rate = generate_speaker_audio(
                parler_model, parler_tokenizer, text, speaker1_description
            )
        else:
            audio_arr, rate = generate_speaker_audio(
                parler_model, parler_tokenizer, text, speaker2_description
            )

        # Convert to AudioSegment (pydub will handle sample rate conversion automatically)
        audio_segment = numpy_to_audio_segment(audio_arr, rate)

        # Add to final audio
        if final_audio is None:
            final_audio = audio_segment
        else:
            final_audio += audio_segment

    _dir = union.current_context().working_directory
    output_file = Path(_dir) / "podcast.mp3"
    final_audio.export(
        output_file,
        format="mp3",
        bitrate="192k",
        parameters=["-q:a", "0"],
    )
    return output_file


@union.task(
    # cache=True,
    # cache_version="2",
    container_image=audio_image,
    enable_deck=True,
    requests=union.Resources(gpu="1", mem="8Gi"),
    accelerator=accelerators.A100,
    environment={"TRANSFORMERS_VERBOSITY": "debug"},
)
def generate_podcast(clean_transcript: union.FlyteFile) -> union.FlyteFile:
    with open(clean_transcript, "r") as f:
        podcast_text = json.load(f)

    assert isinstance(podcast_text, list)
    assert isinstance(podcast_text[0], list)
    assert isinstance(podcast_text[0][0], str)
    assert isinstance(podcast_text[0][1], str)

    audio_file = produce_final_audio(podcast_text)
    return union.FlyteFile(str(audio_file))


@union.task(
    container_image=audio_image,
    enable_deck=True,
    deck_fields=[],
    requests=union.Resources(cpu="2", mem="2Gi"),
)
def create_podcast_deck(podcast: union.FlyteFile, clean_transcript: union.FlyteFile):
    from IPython.display import Audio
    import IPython.display as ipd

    podcast.download()

    audio = Audio(podcast.path)
    ipd.display(audio)

    deck = union.Deck(
        name="Generated Podcast",
        html=audio._repr_html_(),
    )

    with open(clean_transcript, "r") as f:
        podcast_text = json.load(f)

    markdown_transcript = "# Podcast Transcript\n\n"

    for speaker, text in podcast_text:
        markdown_transcript += f"- **{speaker}:** {text}\n\n"

    deck.append(MarkdownRenderer().to_html(markdown_transcript))
