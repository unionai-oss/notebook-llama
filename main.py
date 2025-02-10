import json
import streamlit as st
import union


WORKFLOW_NAME = "notebook_llama.pdf_to_podcast.pdf_to_podcast"
UPLOAD_FILE_PATH = "./uploaded_file.pdf"

remote = union.UnionRemote()

if "running_execution_id" not in st.session_state:
    st.session_state["running_execution_id"] = None
if "current_pdf_path" not in st.session_state:
    st.session_state["current_pdf_path"] = None


def generate_podcast(filepath_or_url: str) -> tuple[str, str]:
    workflow = remote.fetch_workflow(
        name=WORKFLOW_NAME,
        project="default",
        domain="development",
    )
    execution = remote.execute(workflow, inputs={"pdf_path": filepath_or_url})
    st.session_state["running_execution_id"] = execution.id.name
    return execution.id.name


def wait_for_execution(execution_id: str):
    execution = remote.fetch_execution(name=execution_id)
    url = remote.generate_console_url(execution)

    n_retries = 360

    node_map = {
        "n0": "Extracting PDF",
        "n1": "Generating Transcript",
        "n2": "Rewriting Transcript",
        "n3": "Generating Podcast",
    }

    with st.status(
        "🚀 Generating podcast",
        expanded=True,
    ) as status:
        st.write(f"Running workflow [here]({url})")
        st.write("This may take about 10 minutes to complete.")
        bar = st.progress(0)

        for _ in range(n_retries):
            execution = remote.sync(execution, sync_nodes=True)

            n_complete_nodes = 0
            for node_name in node_map:
                node = execution.node_executions.get(node_name)
                if node is not None and node.is_done:
                    n_complete_nodes += 1

            prog = (n_complete_nodes + 1) / (len(node_map) + 1)
            text = "" if n_complete_nodes == len(node_map) else f"{node_map[f'n{n_complete_nodes}']}"
            bar.progress(prog, text=text)

            if execution.is_done:
                bar.empty()
                status.update(label="🎙️ Podcast generated!", state="complete", expanded=False)
                st.session_state["running_execution_id"] = None
                break

    podcast_audio_file = execution.outputs["podcast"]
    podcast_audio_file.download()

    transcript_file = execution.outputs["transcript"]
    transcript_file.download()

    return podcast_audio_file.path, transcript_file.path


def main():
    st.title("📖🎙️ NotebookLlama")
    st.write("Powered by [Union](https://union.ai)")
    st.write("Generates a podcast from a PDF.")

    default_url = "https://www.biorxiv.org/content/10.1101/544593v2.full.pdf"
    pdf_url = st.text_input("Enter a PDF URL", value=default_url)
    uploaded_file = st.file_uploader("Or upload a PDF.")
    
    if pdf_url is not None or uploaded_file is not None:
        if uploaded_file is not None:
            pdf_path = UPLOAD_FILE_PATH
        else:
            pdf_path = pdf_url

        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()

            with open(UPLOAD_FILE_PATH, "wb") as f:
                f.write(bytes_data)

            st.write("File uploaded successfully")

        podcast_audio_file, transcript_file = None, None

        # overwrite the current pdf path if it's different
        if st.session_state["current_pdf_path"] != pdf_path:
            st.session_state["current_pdf_path"] = pdf_path

        has_running_execution = (
            st.session_state["running_execution_id"] is not None
            and st.session_state["current_pdf_path"] is not None
            and st.session_state["current_pdf_path"] != pdf_path
        )

        generate_button = st.button("Generate Podcast", type="primary", disabled=has_running_execution)

        if generate_button or has_running_execution:
            if pdf_path is None or pdf_path == "":
                st.error("Please upload a PDF or enter a PDF URL.")
                return
            else:
                execution_id = (
                    st.session_state["running_execution_id"]
                    if has_running_execution
                    else generate_podcast(pdf_path)
                )

            podcast_audio_file, transcript_file = wait_for_execution(execution_id)

        if podcast_audio_file is not None:
            st.audio(podcast_audio_file)

            with open(podcast_audio_file, "rb") as f:
                st.download_button(
                    label="Download Podcast",
                    data=f,
                    file_name="podcast.mp3",
                    mime="audio/mp3",
                )

            with open(transcript_file, "r") as f:
                transcript = json.load(f)

            st.write("## Transcript")
            with st.container(border=True, height=300):
                for speaker, text in transcript:
                    st.write(f"**{speaker}**: {text}")


main()
