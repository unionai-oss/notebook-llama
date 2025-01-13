import io
import json
from pathlib import Path

import union
from flytekit.extras import accelerators

from notebook_llama.images import audio_image

from IPython.display import Audio
import IPython.display as ipd
from tqdm import tqdm

import torch
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

speaker1_description = """
Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise.
"""

bark_sampling_rate = 24000


def create_tts_pipeline_speaker1():
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from parler_tts import ParlerTTSForConditionalGeneration

    bitsandbytes_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        torch_dtype="auto",
        quantization_config=bitsandbytes_config,
    ).to(device)
    parler_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    return parler_model, parler_tokenizer


def create_tts_pipeline_speaker2():
    from transformers import AutoProcessor, BarkModel, BitsAndBytesConfig

    bitsandbytes_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    bark_processor = AutoProcessor.from_pretrained("suno/bark")
    bark_model = (
        BarkModel.from_pretrained(
            "suno/bark",
            torch_dtype="auto",
            quantization_config=bitsandbytes_config,
        )
        .to(device)
        .to_bettertransformer()
    )
    return bark_processor, bark_model


def generate_speaker1_audio(parler_model, parler_tokenizer, text):
    """Generate audio using ParlerTTS for Speaker 1"""
    input_ids = parler_tokenizer(
        speaker1_description, return_tensors="pt"
    ).input_ids.to(device)
    prompt_input_ids = parler_tokenizer(text, return_tensors="pt").input_ids.to(device)
    generation = parler_model.generate(
        input_ids=input_ids, prompt_input_ids=prompt_input_ids
    )
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr, parler_model.config.sampling_rate


def generate_speaker2_audio(bark_model, bark_processor, text):
    """Generate audio using Bark for Speaker 2"""
    inputs = bark_processor(text, voice_preset="v2/en_speaker_6").to(device)
    speech_output = bark_model.generate(
        **inputs, temperature=0.9, semantic_temperature=0.8
    )
    audio_arr = speech_output[0].cpu().numpy()
    return audio_arr, bark_sampling_rate


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

    parler_model, parler_tokenizer = create_tts_pipeline_speaker1()
    bark_processor, bark_model = create_tts_pipeline_speaker2()

    for speaker, text in tqdm(
        podcast_text, desc="Generating podcast segments", unit="segment"
    ):
        if speaker == "Speaker 1":
            audio_arr, rate = generate_speaker1_audio(
                parler_model, parler_tokenizer, text
            )
        else:  # Speaker 2
            audio_arr, rate = generate_speaker2_audio(bark_model, bark_processor, text)

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
    container_image=audio_image,
    enable_deck=True,
    requests=union.Resources(gpu="1", mem="2Gi"),
    accelerator=accelerators.A100,
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
