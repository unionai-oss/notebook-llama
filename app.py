import os

import union
from union.app import App

image = union.ImageSpec(
    name="streamlit-app",
    packages=["streamlit==1.41.1", "union-runtime>=0.1.10", "union==0.1.139"],
)

app = App(
    name="notebook-lm-test-4",
    container_image=image,
    args=[
        "streamlit",
        "run",
        "main.py",
        "--server.port",
        "8080",
    ],
    include=[
        "./main.py",
    ],
    # secrets=[union.Secret(key="huggingface_api_key")],
    env={"UNION_API_KEY": os.environ.get("UNION_API_KEY")},
    port=8080,
    limits=union.Resources(cpu="2", mem="2Gi"),
)
