import ast
import json
from pathlib import Path
import union

from flytekit.deck import MarkdownRenderer
from flytekit.extras import accelerators
from notebook_llama.common import llm_image


N_RETRIES = 5
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """
You are an international oscar winnning screenwriter

You have been working with multiple award winning podcasters.

Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.

Make it as engaging as possible, Speaker 1 and 2 will be simulated by different voice engines

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting.

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the Speaker 2.

REMEMBER THIS WITH YOUR HEART
The TTS Engine for Speaker 1 cannot do "umms, hmms" well so keep it straight text

For Speaker 2 use "umm, hmm" as much, you can also use [sigh] and [laughs]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

Please re-write to make it as characteristic as possible

START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:

STRICTLY RETURN YOUR RESPONSE AS A LIST OF LISTS OK?

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

Example of response:
[
    ["Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."],
    ["Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"],
    ["Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."],
    ["Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?"]
]
"""


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
def rewrite_transcript(transcript: str) -> union.FlyteFile:
    import huggingface_hub

    huggingface_hub.login(
        token=union.current_context().secrets.get(key="huggingface_api_key")
    )
    print(f"System prompt: {SYSTEM_PROMPT}")
    print(f"Transcript: {transcript}")

    pipeline = create_llm_pipeline()
    print(f"Pipeline:\n{pipeline}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]

    response = None
    for i in range(N_RETRIES):
        outputs = pipeline(messages, max_new_tokens=8192, temperature=1)
        print(f"Outputs: {outputs}")
        transcript = outputs[0]["generated_text"][-1]["content"]

        try:
            response = ast.literal_eval(transcript)
            break
        except SyntaxError as e:
            if i == N_RETRIES - 1:
                raise SyntaxError(f"Unable to parse output:\n{outputs}") from e
            print(f"Error: {e}")
            continue

    print(f"Response: {response}")
    union.Deck(
        "Clean transcript",
        MarkdownRenderer().to_html(json.dumps(response, indent=2)),
    )

    _dir = union.current_context().working_directory
    output_file = Path(_dir) / "clean_transcript.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(response, f)

    return union.FlyteFile(str(output_file))
