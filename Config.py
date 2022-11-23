from dataclasses import dataclass

from transformers import AutoTokenizer


@dataclass(frozen=False)
class Config:
    tokenizer: AutoTokenizer
    table_tokenizer: AutoTokenizer
    cell_sep_token = '|'  # TODO
    document_indicator_token = '[unused2]'
    query_indicator_token = '[unused3]'
    task = None

    model_args = None
    model_type: str
    table_model_type: str # although only tapas is used
