from dataclasses import dataclass

from transformers import AutoTokenizer


@dataclass(frozen=False)
class Config:
    tokenizer: AutoTokenizer
