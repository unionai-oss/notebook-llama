from pathlib import Path

import union

from flytekit.deck import MarkdownRenderer
from flytekit.extras import accelerators
from notebook_llama.common import llm_image


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """
You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris.

We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

You have won multiple podcast awards for your writing.
 
Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic.

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting.

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the second speaker.

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:
DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
"""


def read_file_to_string(filename: str) -> str:
    # Try UTF-8 first (most common encoding for text files)
    try:
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        # If UTF-8 fails, try with other common encodings
        encodings = ["latin-1", "cp1252", "iso-8859-1"]
        for encoding in encodings:
            try:
                with open(filename, "r", encoding=encoding) as file:
                    content = file.read()
                print(f"Successfully read file using {encoding} encoding.")
                return content
            except UnicodeDecodeError:
                continue
        raise


def create_llm_pipeline():
    import torch
    import transformers
    from transformers import BitsAndBytesConfig

    pipeline = transformers.pipeline(
        "text-generation",
        model=DEFAULT_MODEL,
        model_kwargs={
            "torch_dtype": "auto",
            "use_safetensors": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.bfloat16,
            ),
        },
        device_map="auto",
    )

    return pipeline


@union.task(
    container_image=llm_image,
    enable_deck=True,
    requests=union.Resources(gpu="1", mem="2Gi"),
    accelerator=accelerators.A100,
    secret_requests=[union.Secret(key="huggingface_api_key")],
)
def write_transcript(pdf_text: union.FlyteFile) -> union.FlyteFile:
    import huggingface_hub

    huggingface_hub.login(
        token=union.current_context().secrets.get(key="huggingface_api_key")
    )
    print(f"System prompt: {SYSTEM_PROMPT}")

    pdf_text.download()
    input_prompt = read_file_to_string(pdf_text.path)
    print(f"Input prompt: {input_prompt}")

    pipeline = create_llm_pipeline()
    print(f"Pipeline:\n{pipeline}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_prompt},
    ]
    outputs = pipeline(messages, max_new_tokens=8192, temperature=1)
    print(f"Outputs: {outputs}")
    response = outputs[0]["generated_text"][-1]["content"]

    union.Deck("Raw transcript", MarkdownRenderer().to_html(response))

    _dir = union.current_context().working_directory
    output_file = Path(_dir) / "raw_transcript.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response)
    return union.FlyteFile(str(output_file))
