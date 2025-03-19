import io
import json
from pathlib import Path

import union
from flytekit.deck import MarkdownRenderer

from notebook_llama.actors import tts_actor, load_kokoro_pipeline


speaker1_description = """
Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise.
"""

speaker2_description = """
Gary's voice is calm and smooth in delivery, speaking at a moderate pace with a very close recording that almost has no background noise.
"""

TTS_MODEL = "parler-tts/parler-tts-mini-v1"
KOKORO_SAMPLE_RATE = 24000


def generate_speaker_audio(parler_model, parler_tokenizer, device, text, speaker_description):
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


def generate_kokoro_speaker_audio(pipeline, text: str, voice: str):
    import torch

    with torch.no_grad():
        generator = pipeline(text, voice=voice, speed=1, split_pattern=r"\n+")
        audio_list = []
        for _, (_, _, audio) in enumerate(generator):
            audio_list.append(audio)
        audio_arr = torch.cat(audio_list).cpu().to(torch.float32).numpy().squeeze()
    return audio_arr, KOKORO_SAMPLE_RATE


def numpy_to_audio_segment(audio_arr, sampling_rate):
    """Convert numpy array to AudioSegment"""
    import numpy as np
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
    import torch
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_audio = None

    kokoro_pipeline = load_kokoro_pipeline(device)
    # parler_model, parler_tokenizer = load_tts_pipeline(TTS_MODEL, device, use_4bit=False)

    for speaker, text in tqdm(
        podcast_text, desc="Generating podcast segments", unit="segment"
    ):
        text = text.replace("\n", " ")
        if speaker in {"Laura", "Speaker 1"} or "Laura" in speaker or "Speaker 1" in speaker:
            audio_arr, rate = generate_kokoro_speaker_audio(
                kokoro_pipeline, text, "af_heart"
            )
        else:
            audio_arr, rate = generate_kokoro_speaker_audio(
                kokoro_pipeline, text, "am_liam"
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


@tts_actor.task(
    cache=True,
    cache_version="5",
    enable_deck=True,
)
def generate_podcast(clean_transcript: union.FlyteFile) -> union.FlyteFile:
    from IPython.display import Audio

    with open(clean_transcript, "r") as f:
        podcast_text = json.load(f)

    assert isinstance(podcast_text, list)
    assert isinstance(podcast_text[0], list)
    assert isinstance(podcast_text[0][0], str)
    assert isinstance(podcast_text[0][1], str)

    audio_file = produce_final_audio(podcast_text)
    audio = Audio(audio_file)
    deck = union.Deck(
        name="Generated Podcast",
        html=audio._repr_html_(),
    )

    markdown_transcript = "# Podcast Transcript\n\n"
    for speaker, text in podcast_text:
        markdown_transcript += f"- **{speaker}:** {text}\n\n"

    deck.append(MarkdownRenderer().to_html(markdown_transcript))
    return union.FlyteFile(str(audio_file))
