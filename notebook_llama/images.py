import union


image = union.ImageSpec(
    name="notebook_llama",
    packages=[
        "PyPDF2>=3.0.0",
        "huggingface-hub>=0.19.0",
        "torch>=2.0.0",
        "transformers>=4.46.0",
        "accelerate>=0.27.0",
        "rich>=13.0.0",
        "ipywidgets>=8.0.0",
        "tqdm>=4.66.0",
    ],
)

llm_image = image.with_packages(["bitsandbytes>=0.41.0"])

audio_image = image.with_apt_packages(["ffmpeg", "git", "llvm"]).with_packages(
    [
        "bitsandbytes>=0.41.0",
        "llvmlite==0.43.0",
        "numba==0.60.0",
        "optimum",
        "bark",
        "parler-tts",
        "pydub",
        "scipy",
    ]
)
