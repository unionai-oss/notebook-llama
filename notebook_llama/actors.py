"""Actor definitions for NotebookLlama workflow."""

import union
from flytekit.extras import accelerators

from notebook_llama.images import audio_image, llm_image


llama_preprocessing_actor = union.ActorEnvironment(
    name="llama-preprocessing-actor",
    container_image=llm_image,
    requests=union.Resources(gpu="1", mem="2Gi"),
    ttl_seconds=300,
    accelerator=accelerators.L4,
    secret_requests=[union.Secret(key="huggingface_api_key")],
    environment={"TRANSFORMERS_VERBOSITY": "debug"},
)


llama_writing_actor = union.ActorEnvironment(
    name="llama-writing-actor",
    container_image=llm_image,
    requests=union.Resources(gpu="1", mem="2Gi"),
    ttl_seconds=300,
    accelerator=accelerators.L4,
    secret_requests=[union.Secret(key="huggingface_api_key")],
    environment={"TRANSFORMERS_VERBOSITY": "debug"},
)


parler_tts_actor = union.ActorEnvironment(
    name="parler-tts-actor",
    container_image=audio_image,
    requests=union.Resources(gpu="1", mem="4Gi"),
    ttl_seconds=300,
    accelerator=accelerators.T4,
    secret_requests=[union.Secret(key="huggingface_api_key")],
    environment={"TRANSFORMERS_VERBOSITY": "debug"},
)


@union.actor_cache
def load_llm_model(model_name: str):
    import torch
    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        # torch_dtype="auto",
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_quant_storage=torch.bfloat16,
        # ),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
    model, tokenizer = accelerator.prepare(model, tokenizer)
    return model, tokenizer


@union.actor_cache
def load_llm_pipeline(model_name: str):
    import torch
    import transformers
    from transformers import BitsAndBytesConfig

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={
            "use_safetensors": True,
            "torch_dtype": torch.bfloat16,
            # "torch_dtype": "auto",
            # "quantization_config": BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_quant_storage=torch.bfloat16,
            # ),
        },
        device_map="auto",
    )

    return pipeline


@union.actor_cache
def load_tts_pipeline(model_name: str, device: str, use_4bit: bool = False):
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from parler_tts import ParlerTTSForConditionalGeneration

    kwargs = {}
    if use_4bit:
        bitsandbytes_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        kwargs["quantization_config"] = bitsandbytes_config
        kwargs["torch_dtype"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name, **kwargs,
    ).to(device)
    parler_tokenizer = AutoTokenizer.from_pretrained(model_name)
    return parler_model, parler_tokenizer


@union.actor_cache
def load_kokoro_pipeline(device: str):
    import torch
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a", device=device)
    # pipeline.model = pipeline.model.to(dtype=torch.bfloat16)
    return pipeline
