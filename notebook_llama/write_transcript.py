from pathlib import Path

import union

from flytekit.deck import MarkdownRenderer
from notebook_llama.actors import llama_writing_actor, load_llm_pipeline


DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_3_1_8B_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """
You are the a world-class podcast writer, you have worked with many of the top podcasters in the world, like Ira Glass, Stephen J. Dubner, and many more.

We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

You have won multiple podcast awards for your writing.
 
Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic.

Remember Speaker 2 is an expert on the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and asks pointed questions to the speaker 2 and keeps the conversation on track by asking follow up questions. Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Speaker 2: Is an expert on the topic at hand, gives incredible anecdotes and analogies when explaining complex and abstract concepts. Is a captivating teacher that gives great anecdotes

Make sure the tangents speaker 2 provides are quite wild or interesting.

Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:
DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
MAKE SURE TO UNPACK ACRONYMS AND TECHNICAL TERMS FOR LAY AUDIENCES
DON'T INCLUDE ANY TEXT THAT INDICATES THE SPEAKER'S TONE, for example: excitedly, laughs, pauses, clears throat, gets excited

Example of response:

Speaker 1: Welcome to our podcast where we explore the latest advancements in AI and technology. I'm your host, Laura, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI.

Speaker 2: Hi, I'm Liam, and I'm excited to be here! So, what is Llama 3.2?

Speaker 1: Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options.

Speaker 2: That sounds amazing! What are some of the key features of Llama 3.2?
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


@llama_writing_actor.task(
    cache=True,
    cache_version="3",
    enable_deck=True,
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

    print("Loading pipeline")
    pipeline = load_llm_pipeline(DEFAULT_MODEL)
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
