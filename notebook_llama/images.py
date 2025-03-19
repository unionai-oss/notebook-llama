import union


llm_image = union.ImageSpec(
    name="notebook_llama_image_v0",
    packages=[
        "requests==2.32.3",
        "PyPDF2>=3.0.0",
        "huggingface-hub>=0.19.0",
        "torch>=2.0.0",
        "transformers>=4.46.0",
        "accelerate>=0.27.0",
        "rich>=13.0.0",
        "ipywidgets>=8.0.0",
        "tqdm>=4.66.0",
        "union==0.1.162",
        "flytekit==1.15.3",
    ],
)

audio_image = llm_image.with_apt_packages(
    ["espeak-ng", "ffmpeg", "git", "llvm"]
).with_packages(
    [
        "llvmlite==0.43.0",
        "kokoro>=0.3.4",
        "numba==0.60.0",
        "optimum",
        "pydub",
        "scipy",
        "soundfile",
        "spacy",
    ]
).with_commands(
    [
        "python -m spacy download en_core_web_sm",
    ]
)
