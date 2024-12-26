import union


image = union.ImageSpec(
    name="notebook_llama",
    packages=[
        "PyPDF2>=3.0.0",
        "torch>=2.0.0",
        "transformers>=4.46.0",
        "accelerate>=0.27.0",
        "rich>=13.0.0",
        "ipywidgets>=8.0.0",
        "tqdm>=4.66.0",
    ],
)
