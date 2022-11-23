from typing import List, Tuple, Dict
import re
import numpy as np
from rank_bm25 import BM25Okapi


def get_bm25(tokenizer, corpus, query):
    tokenized_corpus = [tokenizer.tokenize(doc) for doc in corpus]
    tokenized_query = tokenizer.tokenize(query)
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    results = []
    for i, text in enumerate(corpus):
        results.append(
            {
                'index': i,
                'text': text,
                'score': scores[i]
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


def table_identify(table):
    # 1:行表头和列表头均有;2.只有行表头;3.只有列表头;4.其他(暂时不启用)
    money = re.compile("^\$ [-+]?[0-9]*\.?[0-9]+")
    number = re.compile("\$ [-+]?[0-9]*\.?[0-9]+")
    table_class = 1
    if re.findall(money, table[0][-1]):
        table_class = 2
    count = 0
    for j in range(len(table[-1])):
        if re.findall(number, table[-1][j]):
            count += 1
    if count == len(table[-1]):
        table_class = 3
    return table, table_class


def cell_to_sentence(table, table_class, x, y):
    def remove_space(text_in):
        res = []
        for tmp in text_in.split(" "):
            if tmp != "":
                res.append(tmp)
        return " ".join(res)

    text = ""
    if table[x][y] == "NULL":
        return text
    if table_class == 1:
        if x == 0:
            text = table[x][y]  # text = "the "+table[0][y]+" is "+table[-1][y]+" ; "
        elif y == 0:
            text = table[x][y]  # text = "the "+table[x][0]+" is "+table[x][-1]+" ; "
        else:
            text = "the " + table[x][0] + " " + table[0][y] + " is " + table[x][y] + " ; "
    elif table_class == 2:
        if y == 0:
            text = table[x][y]  # text = "the "+table[x][0]+" is "+table[x][-1]+" ; "
        else:
            text = "the " + table[x][0] + " is " + table[x][y] + " ; "
    elif table_class == 3:
        if x == 0:
            text = table[x][y]  # text = "the "+table[0][y]+" is "+table[-1][y]+" ; "
        else:
            text = "the " + table[0][y] + " is " + table[x][y] + " ; "
    else:
        text = table[x][y]
    return remove_space(text).strip()


def fin_grained_retriever(preds: List[float], M: int, N: int, topn: int) -> Dict:
    '''
    :param preds: 浮点数数组。模型预测的分数，从前往后分别为pre_texts, post_texts, MxN个cells的相关分数。数组长度为
                    len(pre_texts) + len(post_texts) + MxN
    :param M: table行的大小
    :param N: table列的大小
    :param topn: 选择前topn个数据作为搜索结果
    :return:返回降序搜索结果字典。字典key：text或cell；字典value：搜索结果。
    '''
    text_preds = preds[:(-1) * M * N]
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
    retrieved = {
        "text": [],
        "cell": []
    }
    for dict in sorted_dicts[:topn]:
        if "text" in dict:
            retrieved["text"].append(dict["text"])
        elif "cell" in dict:
            retrieved["cell"].append(dict["cell"])
    return retrieved


def fin_grained_retrieval_metrics(preds: List[float], M: int, N: int, text_labels: List[int],
                                  cell_labels: List[List[int]]) -> Dict:
    '''
    :param preds: 浮点数数组。模型预测的分数，从前往后分别为pre_texts, post_texts, MxN个cells的相关分数。数组长度为
                    len(pre_texts) + len(post_texts) + MxN
    :param M: table行的大小
    :param N: table列的大小
    :param text_labels: 整型数数组。Gold Texts在pre_texts+post_texts中的下标
    :param cell_labels: 位置坐标数组。Gold Cells的坐标
    :return: 返回搜索指标字典。字典key：指标名字字符串；字典value：指标分数浮点数。
    '''
    topn = 5
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

    gold_number = len(text_labels) + len(cell_labels)
    correct = 0.0
    map = 0.0
    precision = 0.0
    recall = 0.0
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
        if i == topn:
            precision = correct / gold_number
            recall = float(correct == gold_number)
        if correct == 1.0 and mrr == 0.0:
            mrr = 1 / float(i + 1)
    map = map / gold_number
    retrieval_metrics = {
        "precision": precision,
        "recall": recall,
        "MAP": map,
        "MRR": mrr
    }
    return retrieval_metrics


def fin_grained_total_retrieval_metrics(preds: List[List[float]], M: List[int], N: List[int],
                                        text_labels: List[List[int]],
                                        cell_labels: List[List[List[int]]]) -> Dict:
    '''
    :param preds: 浮点数数组。模型预测的分数，从前往后分别为pre_texts, post_texts, MxN个cells的相关分数。数组长度为
                    len(pre_texts) + len(post_texts) + MxN
    :param M: table行的大小
    :param N: table列的大小
    :param text_labels: 整型数数组。Gold Texts在pre_texts+post_texts中的下标
    :param cell_labels: 位置坐标数组。Gold Cells的坐标
    :return: 返回搜索指标字典。字典key：指标名字字符串；字典value：指标分数浮点数。
    '''
    length = len(preds)
    precision = 0.0
    recall = 0.0
    map = 0.0
    mrr = 0.0
    for i in range(length):
        retrieval_metrics = fin_grained_retrieval_metrics(preds[i], M[i], N[i], text_labels[i], cell_labels[i])
        precision += retrieval_metrics["precision"]
        recall += retrieval_metrics["recall"]
        map += retrieval_metrics["MAP"]
        mrr += retrieval_metrics["MRR"]
    precision = precision / length
    recall = recall / length
    map = map / length
    mrr = mrr / length
    retrieval_metrics = {
        "precision": precision,
        "recall": recall,
        "MAP": map,
        "MRR": mrr
    }
    return retrieval_metrics
