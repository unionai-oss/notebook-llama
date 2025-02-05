import json
from datetime import timedelta
import streamlit as st
import union


WORKFLOW_NAME = "notebook_llama.pdf_to_podcast.pdf_to_podcast"
UPLOAD_FILE_PATH = "./uploaded_file.pdf"


@st.cache_data(show_spinner=False)
def generate_podcast(filepath_or_url: str):
    remote = union.UnionRemote()

    workflow = remote.fetch_workflow(
        name=WORKFLOW_NAME,
        project="default",
        domain="development",
    )
    execution = remote.execute(workflow, inputs={"pdf_path": filepath_or_url})
    url = remote.generate_console_url(execution)

    with st.spinner(
        f"Generating podcast: [go to workflow execution]({url})\n\n"
        "Hold tight! For uncached PDFs, this will take about 20 minutes to complete."
    ):
        execution = remote.wait(execution, poll_interval=timedelta(seconds=5))

    podcast_audio_file = execution.outputs["podcast"]
    podcast_audio_file.download()

    transcript_file = execution.outputs["transcript"]
    transcript_file.download()

    return podcast_audio_file.path, transcript_file.path


def main():
    st.title("📖🎙️ My NotebookLM Clone")
    st.write("This app generates a podcast from a PDF.")

    default_url = "https://www.biorxiv.org/content/10.1101/544593v2.full.pdf"
    pdf_url = st.text_input("Enter a PDF URL", value=default_url)
    uploaded_file = st.file_uploader("Or upload a PDF.")
    
    if pdf_url is not None or uploaded_file is not None:
        pdf_path = pdf_url if pdf_url is not None else UPLOAD_FILE_PATH

        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()

            with open(UPLOAD_FILE_PATH, "wb") as f:
                f.write(bytes_data)

            st.write("File uploaded successfully")

        podcast_audio_file, transcript_file = None, None
        if st.button("Generate Podcast", type="primary"):
            podcast_audio_file, transcript_file = generate_podcast(pdf_path)

        if podcast_audio_file is not None:
            st.audio(podcast_audio_file)

            with open(transcript_file, "r") as f:
                transcript = json.load(f)

            st.write("## Transcript")
            with st.container(border=True, height=300):
                for speaker, text in transcript:
                    st.write(f"**{speaker}**: {text}")


main()

# TODO:
# - Return execution id with console url
# - Add "Get podcast" button to load the podcast. If execution is not complete,
#   show spinner