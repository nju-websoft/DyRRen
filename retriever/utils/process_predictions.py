from cProfile import label
import json
import numpy as np
from typing import List, Tuple, Dict

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


def process_data(data_type, topn = 5):
    data_dir = 'FinQADataset/'
    data_file = data_dir + f'retriever_{data_type}.json'
    preds_file = data_dir + f'preds_{data_type}.json'
    out_file = data_dir + f'generator_{data_type}.json'
    with open(data_file) as f_obj:
        datas = json.load(f_obj)
    with open(preds_file) as f_obj:
        preds=json.load(f_obj)
    
    all_preds_scores = []
    all_texts_labels = []

    for pred in preds:
        index = pred[0][0]
        text_labels = pred[1]
        scores = pred[2]
        data = datas[index]

        all_preds_scores.append(scores)
        all_texts_labels.append(text_labels)

        if data['index'] != index:
            print(data['id'])

        texts = data['texts']
        text_cell_labels = data['labels']
        gold_texts = [texts[label] for label in text_labels]
        rank_texts = []
        for text, label, score in zip(texts, text_cell_labels, scores):
            type = 'table' if label == 1 else 'document'
            rank_texts.append({
                'text':text,
                'score':score,
                'type':type
            })
        rank_texts = sorted(rank_texts, key=lambda kv: kv['score'], reverse=True)

        model_input = []
        if data_type != 'train':
            model_input = [x['text'] for x in rank_texts[:topn]]
        else:
            model_input = [x['text'] for x in rank_texts[:topn]]
            for gold_text in gold_texts:
                if gold_text not in model_input:
                    model_input.append(gold_text)

        qa=data['qa']

        if data_type != 'private_test':
            del qa['explanation']
            del qa['ann_table_rows']
            del qa['ann_text_rows']
            del qa['tfidftopn']
            del qa['model_input']
            del qa['program_re']
            qa['gold_input'] = gold_texts
        
        qa['model_input'] = model_input
        del data['index']
        del data['labels']

        data['qa'] = qa
        data['texts'] = rank_texts

        datas[index] = data
    
    if data_type != 'private_test':
        results = fin_grained_total_retrieval_metrics(preds=all_preds_scores, M=[0] * len(preds), N=[0] * len(preds),
                                                text_labels=all_texts_labels, cell_labels=[[]] * len(preds),topn=topn)
        print(results)
    print(f'process {data_type} done!')
    with open(out_file,'w',encoding='utf-8') as f_obj:
            json.dump(datas, f_obj, sort_keys=False, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    topn = 3
    print(topn)
    process_data('train',topn)
    process_data('dev',topn)
    process_data('test',topn)
    # process_data('private_test',topn)
