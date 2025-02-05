import ast
import json
from pathlib import Path
import union

from flytekit.deck import MarkdownRenderer
from notebook_llama.actors import llama_writing_actor, load_llm_pipeline


N_RETRIES = 30
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# DEFAULT_MODEL = "meta-llama/Llama-3.1-70B-Instruct"

SYSTEM_PROMPT = """
You are an international oscar winning screenwriter

You have been working with multiple award winning podcasters.
Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.
Make it as engaging as possible, Speaker 1 and 2 will be simulated by different voice engines

Remember Speaker 2 is an expert on the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and asks pointed questions to the speaker 2 and keeps the conversation on track by asking follow up questions. Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Speaker 2: Is an expert on the topic at hand, gives incredible anecdotes and analogies when explaining complex and abstract concepts. Is a captivating teacher that gives great anecdotes

Make sure the tangents speaker 2 provides are quite wild or interesting.

Ensure there are interruptions during explanations injected throughout from the Speaker 1.

MAKE SURE TO DO THE FOLLOWING:
- The TTS Engine for Speaker 1 and 2 can handle "umms, hmms" so inject them in the text where appropriate
- The TTS Engine also cannot handle emoting annotation, so EXCLUDE any other meta text like (excitedly), (laughs), or (pauses)
- It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait
- The first element in the list of lists should be "Speaker 1" or "Speaker 2"
- The second element in the list of lists should be what the speaker is saying. Make sure to replace [Speaker 1] with Laura and [Speaker 2] with Liam
- Please re-write to make it as characteristic as possible
- START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:
- STRICTLY RETURN YOUR RESPONSE AS A LIST OF LISTS OK?
- IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

Example of response:
[
    ["Speaker 1", "Welcome to our podcast where we explore the latest advancements in AI and technology. I'm your host, Laura, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."],
    ["Speaker 2", "Hi, I'm Liam, and I'm excited to be here! So, what is Llama 3.2?"],
    ["Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."],
    ["Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?"]
]
"""


@llama_writing_actor.task(
    cache=True,
    cache_version="7",
    enable_deck=True,
)
def rewrite_transcript(transcript: union.FlyteFile) -> union.FlyteFile:
    import huggingface_hub

    huggingface_hub.login(
        token=union.current_context().secrets.get(key="huggingface_api_key")
    )

    with open(transcript, "r", encoding="utf-8") as f:
        transcript = f.read()

    print(f"[System prompt]\n\n{SYSTEM_PROMPT}")
    print(f"[Transcript]\n\n{transcript}")

    print("Loading pipeline")
    pipeline = load_llm_pipeline(DEFAULT_MODEL)
    print(f"Pipeline:\n{pipeline}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]

    response = None
    for i in range(N_RETRIES):
        outputs = pipeline(messages, max_new_tokens=8192, temperature=1)
        print(f"[Outputs]\n\n{outputs}")
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
