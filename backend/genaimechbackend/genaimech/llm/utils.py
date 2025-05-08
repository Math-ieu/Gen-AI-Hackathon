from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import yaml
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_prompt_medical(question, context, answer=None):
    """Generates a prompt from the given question, context, and answer."""
    if answer:
        return f"question: {question} context: {context} answer: {answer} </s>"
    else:
        return f"question: {question} context: {context} </s>"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5")

query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)

# Global variables for lazy loading
_model = None
_tokenizer = None
_eval_tokenizer = None

def load_models():
    global _model, _tokenizer, _eval_tokenizer
    if _model is None:
        # Load tokenizer for the model
        x = "distilgpt2"
        _tokenizer = AutoTokenizer.from_pretrained(x)
        _tokenizer.pad_token = _tokenizer.eos_token

        # Load model (non-quantized, CPU, float16)
        _model = AutoModelForCausalLM.from_pretrained(
            x,
            device_map="cpu",
            torch_dtype=torch.float16,  # Use float16 to reduce memory
            use_cache=False
        )

        # Load tokenizer for the base model (same as main tokenizer)
        _eval_tokenizer = AutoTokenizer.from_pretrained(
            x,
            add_bos_token=True,
            trust_remote_code=True
        )

    return _model, _model, _tokenizer, _eval_tokenizer  # Reuse _model for _base_model

# Initialize LLM with lazy-loaded model and tokenizer
def get_llm():
    model, _, tokenizer, _ = load_models()
    return HuggingFaceLLM(
        context_window=2048,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.25, "do_sample": False},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer=tokenizer,
        model=model,
        device_map="cpu",
        tokenizer_kwargs={"max_length": 2048}
    )

# Initialize fine-tuned model with lazy-loaded base model
def get_ft_model():
    base_model, _, _, eval_tokenizer = load_models()
    # Note: Checkpoint may not be compatible; using base model for now
    return base_model, eval_tokenizer

# Set global settings
Settings.chunk_size = 512
Settings.llm = get_llm()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import sys
# import os
# from llama_index.llms.gemini.base import Gemini
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv, find_dotenv
# from llama_index.embeddings.gemini import GeminiEmbedding
# import google.generativeai as genai
# import yaml
# from llama_index.core import Settings


# def load_config(CONFIG_PATH):
#     with open(CONFIG_PATH, 'r') as f:
#         config = yaml.safe_load(f)
#     return config


# def generate_prompt_medical(question, context, answer=None):
#     """Generates a prompt from the given question, context, and answer."""
#     if answer:
#         return f"question: {question} context: {context} answer: {answer} </s>"
#     else:
#         return f"question: {question} context: {context} </s>"


# sys.path.append(os.getcwd())

# load_dotenv(find_dotenv())
# genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# llm = ChatGoogleGenerativeAI(model="gemini-pro")

# print("os.environ['GOOGLE_API_KEY']", os.environ['GOOGLE_API_KEY'])

# Settings.embed_model = GeminiEmbedding(
#     model_name="models/embedding-001", api_key=os.environ["GOOGLE_API_KEY"]
# )

# Settings.llm = Gemini()
