from typing import List, Tuple, Dict
import re
import numpy as np
from rank_bm25 import BM25Okapi

def get_bm25(tokenizer ,corpus, query):
    tokenized_corpus = [tokenizer.tokenize(doc) for doc in corpus]
    tokenized_query = tokenizer.tokenize(query)
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    results = []
    for i,text in enumerate(corpus):
        results.append(
            {
                'index':i,
                'text':text,
                'score':scores[i]
            }
        )
    results = sorted(results, key=lambda kv: kv['score'], reverse=True)
    return results

def flatten_table(table: List[List[str]], table_class) -> str:
    table_flatten_str = ''
    table_flatten_str += f'Col | {" | ".join(table[0])} | '
    for row_index, row in enumerate(table[1:]):
        table_flatten_str += f'Row | {row_index} | {" | ".join(row)} | '
    return table_flatten_str


def table_identify(table: List[List[str]]) -> Tuple[List[List[str]], int]:
    money = re.compile("^\$ [-+]?[0-9]*\.?[0-9]+")
    if re.findall(money, table[0][-1]):
        return table, 2
    else:
        return table, 1


def fin_grained_retrieval_metrics(preds: List[float], M: int, N: int, text_labels: List[int],
                                  cell_labels: List[List[int]],topn = 5) -> Dict:
    gold_number = len(text_labels) + len(cell_labels)
    if  gold_number == 0:
        retrieval_metrics = {
            "recall": 0,
            "crsr": 0,
            "MAP": 0,
            "MRR": 0
        }
        return retrieval_metrics
    text_preds = preds[:(-1) * M * N] if M * N != 0 else preds
    cell_preds = np.resize(preds[(-1) * M * N:], (M, N))
    preds_dicts = []
    for i in range(len(text_preds)):
        preds_dicts.append({
            "text": i,
            "score": text_preds[i]
        })
    for i in range(M):
        for j in range(N):
            preds_dicts.append({
                "cell": [i, j],
                "score": cell_preds[i][j]
            })
    sorted_dicts = sorted(preds_dicts, key=lambda kv: kv["score"], reverse=True)

    correct = 0.0
    map = 0.0
    recall = 0.0
    crsr = 0.0
    mrr = 0.0
    for i in range(len(sorted_dicts)):
        if "text" in sorted_dicts[i]:
            if sorted_dicts[i]["text"] in text_labels:
                correct += 1
                map += correct / float(i + 1)
        elif "cell" in sorted_dicts[i]:
            if sorted_dicts[i]["cell"] in cell_labels:
                correct += 1
                map += correct / float(i + 1)
        if i == topn-1 or (topn >= len(sorted_dicts) and i == len(sorted_dicts)-1):
            recall = correct / gold_number
            crsr = float(correct == gold_number)
        if correct == 1.0 and mrr == 0.0:
            mrr = 1 / float(i + 1)
    map = map / gold_number
    retrieval_metrics = {
        "recall": recall,
        "crsr": crsr,
        "MAP": map,
        "MRR": mrr
    }
    return retrieval_metrics


def fin_grained_total_retrieval_metrics(preds: List[List[float]], M: List[int], N: List[int],
                                        text_labels: List[List[int]],
                                        cell_labels: List[List[List[int]]],topn = 5) -> Dict:
    length = len(preds)
    recall = 0.0
    crsr = 0.0
    map = 0.0
    mrr = 0.0
    for i in range(length):
        retrieval_metrics = fin_grained_retrieval_metrics(preds[i], M[i], N[i], text_labels[i], cell_labels[i], topn)
        recall += retrieval_metrics["recall"]
        crsr += retrieval_metrics["crsr"]
        map += retrieval_metrics["MAP"]
        mrr += retrieval_metrics["MRR"]
    recall = recall / length
    crsr = crsr / length
    map = map / length
    mrr = mrr / length
    retrieval_metrics = {
        "recall": recall,
        "crsr": crsr,
        "MAP": map,
        "MRR": mrr
    }
    return retrieval_metrics
