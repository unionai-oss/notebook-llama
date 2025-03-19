from datetime import timedelta

import union
from union.app import App

image = union.ImageSpec(
    name="streamlit-app",
    apt_packages=["git"],
    packages=["streamlit==1.41.1", "union-runtime>=0.1.10", "union>=0.1.160"],
    builder="union",
)

app = App(
    name="notebook-llama-streamlit",
    container_image=image,
    args="streamlit run main.py --server.port 8080 "
        "--server.enableXsrfProtection false "
        "--browser.gatherUsageStats false",
    include=[
        "./main.py",
    ],
    secrets=[union.Secret(key="union_api_key", env_var="UNION_API_KEY")],
    port=8080,
    limits=union.Resources(cpu="2", mem="2Gi"),
    scaledown_after=timedelta(minutes=15),
    requires_auth=False,
)
