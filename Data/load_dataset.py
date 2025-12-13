"""load_dataset.py

Helper for downloading the Arabic generated abstracts dataset
from the Hugging Face Hub.
"""

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
import os


def load_hf_dataset(dataset_name: str = "KFUPM-JRCAI/arabic-generated-abstracts"):
    """Load the project dataset from Hugging Face.

    If an ``HF_TOKEN`` environment variable is defined (via .env),
    it is used to authenticate, otherwise the dataset is loaded
    anonymously (works for public datasets).
    """
    # Load environment variables from .env (if present)
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN", "")

    # Authenticate only if a token is actually provided
    if hf_token:
        login(token=hf_token)

    dataset = load_dataset(dataset_name)
    return dataset
