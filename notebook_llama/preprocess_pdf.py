import os
import functools
from pathlib import Path

import union
from flytekit.deck import MarkdownRenderer
from notebook_llama.images import llm_image
from notebook_llama.actors import llama_preprocessing_actor, load_llm_model


DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

SYS_PROMPT = """
You are a world class text pre-processor, here is the raw data from a PDF, please parse and return it in a way that is crispy and usable to send to a podcast writer.

The raw data is messed up with new lines, Latex math and you will see fluff that we can remove completely. Basically take away any details that you think might be useless in a podcast author's transcript.

Remember, the podcast could be on any topic whatsoever so the issues listed above are not exhaustive

Please be smart with what you remove and be creative ok?

Remember DO NOT START SUMMARIZING THIS, YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED

Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.

Be very smart and aggressive with removing repeating numbers, repeating text, and other text that may have been part of a table or equation.

PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPATILISATION ETC LIKES

ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
Here is the text:
"""


def validate_pdf(file_path: str) -> bool:
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return False
    return True


def extract_text_from_pdf(
    file_path: str | Path,
    max_chars: int = 100000,
) -> str | None:
    import PyPDF2

    if not validate_pdf(str(file_path)):
        return None

    try:
        with open(file_path, "rb") as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)

            # Get total number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Processing PDF with {num_pages} pages...")

            extracted_text = []
            total_chars = 0

            # Iterate through all pages
            for page_num in range(num_pages):
                # Extract text from page
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                # Check if adding this page's text would exceed the limit
                if total_chars + len(text) > max_chars:
                    # Only add text up to the limit
                    remaining_chars = max_chars - total_chars
                    extracted_text.append(text[:remaining_chars])
                    print(f"Reached {max_chars} character limit at page {page_num + 1}")
                    break

                extracted_text.append(text)
                total_chars += len(text)
                print(f"Processed page {page_num + 1}/{num_pages}")

            final_text = "\n".join(extracted_text)
            print(f"\nExtraction complete! Total characters: {len(final_text)}")
            return final_text

    except PyPDF2.PdfReadError:
        print("Error: Invalid or corrupted PDF file")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None


def get_pdf_metadata(file_path: str | Path) -> dict | None:
    import PyPDF2

    if not validate_pdf(file_path):
        return None

    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = {
                "num_pages": len(pdf_reader.pages),
                "metadata": pdf_reader.metadata,
            }
            return metadata
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return None


def create_word_bounded_chunks(text: str, target_chunk_size: int = 10000) -> list[str]:
    """
    Split text into chunks at word boundaries close to the target chunk size.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for the space
        if current_length + word_length > target_chunk_size and current_chunk:
            # Join the current chunk and add it to chunks
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_chunk(
    tokenizer,
    model,
    device,
    text_chunk: str,
    chunk_num: int,
    max_new_tokens: int = 512,
) -> str:
    """Process a chunk of text and return both input and output for verification"""
    import torch

    conversation = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": text_chunk},
    ]

    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
        )

    processed_text = tokenizer.decode(output[0], skip_special_tokens=True)[
        len(prompt) :
    ].strip()

    # Print chunk information for monitoring
    print(f"\n{'='*40} Chunk {chunk_num} {'='*40}")
    print(f"INPUT TEXT:\n{text_chunk[:500]}...")  # Show first 500 chars of input
    print(
        f"\nPROCESSED TEXT:\n{processed_text[:500]}..."
    )  # Show first 500 chars of output
    print(f"{'='*90}\n")

    return processed_text


@union.task(
    container_image=llm_image,
    enable_deck=True,
    cache=True,
    cache_version="13",
    requests=union.Resources(cpu="2", mem="4Gi"),
)
def extract_text(pdf_path: union.FlyteFile) -> union.FlyteFile:

    print(f"Downloading PDF from FlyteFile {pdf_path}")
    pdf_path.download()
    pdf_path = pdf_path.path

    # Extract metadata first
    print("Extracting metadata...")
    metadata = get_pdf_metadata(pdf_path)
    if metadata:
        print("\nPDF Metadata:")
        print(f"Number of pages: {metadata['num_pages']}")
        print("Document info:")
        for key, value in metadata["metadata"].items():
            print(f"{key}: {value}")

    # Extract text
    print("\nExtracting text...")
    extracted_text = extract_text_from_pdf(pdf_path)

    # Display first 500 characters of extracted text as preview
    if extracted_text:
        print("\nPreview of extracted text (first 500 characters):")
        print("-" * 50)
        print(extracted_text[:500])
        print("-" * 50)
        print(f"\nTotal characters extracted: {len(extracted_text)}")

    # Optional: Save the extracted text to a file
    extracted_text = extracted_text or "No text extracted"
    _dir = union.current_context().working_directory
    output_file = Path(_dir) / "extracted_text.txt"
    with open(output_file, "w", encoding="utf-8", errors="ignore") as f:
        f.write(extracted_text)
    print(f"\nExtracted text has been saved to {output_file}")
    union.Deck("Extracted Text", MarkdownRenderer().to_html(extracted_text))
    return union.FlyteFile(str(output_file))


@llama_preprocessing_actor.task(
    cache=True,
    cache_version="6",
    enable_deck=True,
)
def preprocess_pdf(
    input_file: union.FlyteFile,
    chunk_size: int = 16384,
    max_new_tokens: int = 8192,
) -> union.FlyteFile:
    import huggingface_hub
    import torch
    from tqdm import tqdm

    # Read the file
    input_file.download()
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()

    huggingface_hub.login(
        token=union.current_context().secrets.get(key="huggingface_api_key")
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, tokenizer = load_llm_model(DEFAULT_MODEL)
    print(f"Model: {model}")
    print(f"Tokenizer: {tokenizer}")

    # Calculate number of chunks
    num_chunks = (len(text) + chunk_size - 1) // chunk_size

    # Create output file name
    _dir = union.current_context().working_directory
    output_file = Path(_dir) / f"clean_{os.path.basename(input_file.path)}"

    chunks = create_word_bounded_chunks(text, chunk_size)
    num_chunks = len(chunks)

    _process_chunk = functools.partial(process_chunk, tokenizer, model, device)
    processed_text = ""

    with open(output_file, "w", encoding="utf-8") as out_file:
        for chunk_num, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            # Process chunk and append to complete text
            processed_chunk = _process_chunk(chunk, chunk_num, max_new_tokens)
            processed_text += processed_chunk + "\n"

            # Write chunk immediately to file
            out_file.write(processed_chunk + "\n")
            out_file.flush()

    print(f"\nProcessing complete!")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Total chunks processed: {num_chunks}")

    union.Deck(
        "Processed Text",
        MarkdownRenderer().to_html(
            "BEGINNING:\n"
            + processed_text[:1000]
            + "\n...\n\nEND:\n"
            + processed_text[-1000:]
        ),
    )

    return union.FlyteFile(str(output_file))


@union.workflow
def pdf_to_text(pdf_path: union.FlyteFile) -> union.FlyteFile:
    extracted_text = extract_text(pdf_path)
    processed_text = preprocess_pdf(extracted_text)
    return processed_text
