# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """

import json
import logging
import os
import random
from dataclasses import dataclass
from enum import Enum
from functools import partial
from math import ceil
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from filelock import FileLock
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, RobertaTokenizer, DataProcessor

from utils.utils import flatten_table, get_bm25, table_identify

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    private_test = "private_test"


@dataclass(frozen=True)
class InputExampleFinQA:
    example_id: str
    texts: List[str]
    table: List[List[str]]
    question: str
    answer: str
    program: List[Dict[str, str]]
    text_label: List[int]
    row_label: List[int]
    cell_label: List[int]


@dataclass(frozen=True)
class InputFeaturesFinQA:
    example_id: str
    table_size: List[int]  # M rows x N columns
    texts_input_ids: List[int]
    table_input_ids: List[int]
    question_input_ids: List[int]
    texts_attention_mask: List[int]
    table_attention_mask: List[int]
    question_attention_mask: List[int]
    texts_token_type_ids: List[int]
    table_token_type_ids: List[int]
    question_token_type_ids: List[int]
    table_cell_intervals: List[List[int]]
    texts_num: int
    text_label: List[int]
    row_label: List[int]
    cell_label: List[int]


class ProcessorFinQA(DataProcessor):

    def __init__(self, max_texts_training_retrieval, max_texts_evaluating_retrieval):
        self.max_texts_training_retrieval = max_texts_training_retrieval
        self.max_texts_evaluating_retrieval = max_texts_evaluating_retrieval

    def get_train_examples(self, data_dir, tokenizer):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(f'{data_dir}/train.json', "train", tokenizer)

    def get_dev_examples(self, data_dir, tokenizer):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(f'{data_dir}/dev.json', "dev", tokenizer)

    def get_test_examples(self, data_dir, tokenizer):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(f'{data_dir}/test.json', "test", tokenizer)
    def get_private_test_examples(self, data_dir, tokenizer):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(f'{data_dir}/private_test.json', "private_test", tokenizer)

    def _read_json(self, input_dir):
        return json.load(open(input_dir, 'r', encoding='utf-8'))

    def _text_filter(self, s):
        return s.strip() not in ['', '.']

    def _table_cell_2_sentence(self, table: List[List[str]]):

        def remove_space(text_in):
            res = []
            for tmp in text_in.split(" "):
                if tmp != "":
                    res.append(tmp)
            return " ".join(res)

        def table_row_to_text(header, row):
            '''
            use templates to convert table row to text
            '''
            import re
            money=re.compile("^\$ [-+]?[0-9]*\.?[0-9]+")
            res = ""

            if re.findall(money,header[-1]):
                for cell in row[1:]:
                    res += ("the " + row[0] + " is " + cell + " ; ")
            else:
                for head, cell in zip(header[1:], row[1:]):
                    res += ("the " + row[0] + " of " + head + " is " + cell + " ; ")

            res = remove_space(res)
            res = res.strip()
            return res

        row_sentences = [table_row_to_text(table[0], table[i]) for i in range(len(table))]
        return row_sentences

    def _create_examples(self, data_dir, data_type, tokenizer) -> List[InputExampleFinQA]:
        """Creates examples for the training and dev sets."""
        examples = []
        fin_datas = []

        raw_datas = self._read_json(data_dir)
        for data_index, data in tqdm(enumerate(raw_datas), total=len(raw_datas),
                                     desc=f'Creating {data_type} examples'):
            example_id = data_index
            pre_text = data['pre_text']
            post_text = data['post_text']
            table = data['table']
            question = data['qa']['question']
            row_sentences = self._table_cell_2_sentence(table)
            assert len(row_sentences) == len(table)
            texts = pre_text + post_text + row_sentences
            texts = list(filter(self._text_filter, texts))

            if data_type != 'private_test':
                gold_texts = list(data['qa']['gold_inds'].values())
                gold_texts = list(set(gold_texts))
                other_texts = list(filter(lambda x: x not in gold_texts, texts))
                bm25_results = get_bm25(tokenizer, other_texts, question)
                other_texts = [x['text'] for x in bm25_results]
                texts = (gold_texts + other_texts)[
                        :self.max_texts_training_retrieval if data_type == 'train' else self.max_texts_evaluating_retrieval]
                random.shuffle(texts)
                text_label = [1 if texts[i] in gold_texts else 0 for i in range(len(texts))]
                text_cell_label = [1 if texts[i] in row_sentences else 0 for i in range(len(texts))]
                assert sum(text_label) > 0, data['id']
            else :
                random.shuffle(texts)
                text_label = text_cell_label = [1 if texts[i] in row_sentences else 0 for i in range(len(texts))]
            examples.append(InputExampleFinQA(example_id=example_id,
                                              texts=texts,
                                              table=table,
                                              question=question,
                                              answer=None,
                                              program=None,
                                              text_label=text_label,
                                              row_label=None,
                                              cell_label=None, ))
            fin_datas.append(
                {
                    'index':example_id,
                    'id':data['id'],
                    'table':data['table'],
                    'texts':texts,
                    'labels':text_cell_label,
                    'qa':data['qa']
                }
            )
        with open(f'FinQADataset/retriever_{data_type}.json','w',encoding='utf-8') as f_obj:
            json.dump(fin_datas, f_obj, sort_keys=False, indent=4, ensure_ascii=False)

        return examples


class DatasetFinQA(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeaturesFinQA]

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            data_args,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        processor = processors[task](data_args.max_texts_training_retrieval, data_args.max_texts_evaluating_retrieval)

        max_seq_length = data_args.max_seq_length
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(max_seq_length),
                os.environ['RUN_NAME'].split('@')[0],
            ),
        )

        logger.info(f'looking for cached file {cached_features_file}')

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        with FileLock(f'{cached_features_file}.lock'):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                # label_list = processor.get_labels()
                if mode == Split.dev:
                    examples = processor.get_dev_examples(data_dir, tokenizer)
                elif mode == Split.test:
                    examples = processor.get_test_examples(data_dir, tokenizer)
                elif mode == Split.train:
                    examples = processor.get_train_examples(data_dir, tokenizer)
                elif mode == Split.private_test:
                    examples = processor.get_private_test_examples(data_dir, tokenizer)
                elif mode == Split.dev_and_test:
                    examples = processor.get_dev_examples(data_dir, tokenizer) + processor.get_test_examples(data_dir, tokenizer)
                else:
                    raise NotImplementedError
                logger.info("Training examples: %s", len(examples))
                self.features = convert_examples_to_features_finqa(
                    examples,
                    max_seq_length,
                    tokenizer,
                    separator_id=tokenizer.encode(' |')[1],
                    filter_id=[tokenizer.encode(' Row')[1]],
                    mode=mode,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeaturesFinQA:
        return self.features[i]


def get_table_cells_location(table_str_ids: List[int], table_size: List[int], separator: int, filter_id: List[id]) -> \
        List[List[int]]:
    """
    :param table_str_ids: input_ids of flatten table string
    :param table_size: table size MxN
    :param separator: separator id
    :return: table cells intervals, size: (MxN) * 2
    """
    separator_indices = list(filter(lambda x: table_str_ids[x] == separator, range(len(table_str_ids))))

    # check. update: do not check due to truncation
    # assert len(separator_indices) == table_size[0] * (table_size[1] + 2) - 1  # M(N+2)-1

    cell_intervals = []

    row_num_flag = False
    for i in range(len(separator_indices) - 1):
        if row_num_flag:
            row_num_flag = False
            continue
        cell_interval = [separator_indices[i] + 1, separator_indices[i + 1]]
        if cell_interval[1] - cell_interval[0] == 1 and table_str_ids[cell_interval[0]] in filter_id:
            row_num_flag = True
            continue
        cell_intervals.append(cell_interval)

    # check. update: do not check due to truncation
    # assert len(cell_intervals) == table_size[0] * table_size[1]

    return cell_intervals


def intervals_2_matrix(intervals: List[List[int]], max_seq_len: int):
    matrix = np.zeros((len(intervals), max_seq_len)).tolist()
    for i, interval in enumerate(intervals):
        for s in range(interval[0], interval[1]):
            matrix[i][s] = 1 / (interval[1] - interval[0])
    return matrix


def convert_examples_to_features_finqa(
        examples: List[InputExampleFinQA],
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        separator_id: int,
        filter_id: List[int],
        mode=Split.train,
) -> List[InputFeaturesFinQA]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    features = []
    trun_count = 0
    total_count = 0
    t = tqdm(enumerate(examples), total=len(examples), desc=f'Converting {mode} examples to features')
    for data_index, example in t:
        table, table_class = table_identify(example.table)
        '''
         ATTENTION: add special token [D] before every text and table, [D]='Document.'
                    add special token [Q] before question, [Q]='Query.'
        '''
        texts = [f'Document. {text}' for text in example.texts]  # [D] text
        question = f'Query. {example.question}'

        tokenized_texts = tokenizer(texts, [question] * len(texts), padding=False, max_length=max_length,  #
                                    truncation=True, return_overflowing_tokens=False, return_length=True)
        tokenized_question = tokenizer([question], padding=False, max_length=max_length,
                                       truncation=True, return_overflowing_tokens=False, return_length=True)

        texts_input_ids = tokenized_texts['input_ids']
        texts_attention_mask = tokenized_texts['attention_mask']
        texts_token_type_ids = tokenized_texts['token_type_ids'] if 'token_type_ids' in tokenized_texts else None

        question_input_ids = tokenized_question['input_ids'][0]
        question_attention_mask = tokenized_question['attention_mask'][0]
        question_token_type_ids = tokenized_question[
            'token_type_ids'][0] if 'token_type_ids' in tokenized_question else None

        trun_count += len(list(filter(lambda x: len(x) >= max_length, texts_input_ids)))
        trun_count += int(len(question_input_ids) >= max_length)
        total_count += len(texts_input_ids) + 1
        t.set_description(f'Converting {mode} examples to features. truncation ratio: {trun_count / total_count}')

        features.append(InputFeaturesFinQA(example_id=example.example_id,
                                           table_size=None,
                                           texts_input_ids=texts_input_ids,
                                           table_input_ids=None,
                                           question_input_ids=question_input_ids,
                                           texts_attention_mask=texts_attention_mask,
                                           table_attention_mask=None,
                                           question_attention_mask=question_attention_mask,
                                           texts_token_type_ids=texts_token_type_ids,
                                           table_token_type_ids=None,
                                           question_token_type_ids=question_token_type_ids,
                                           table_cell_intervals=None,
                                           texts_num=len(texts_input_ids),
                                           text_label=example.text_label,
                                           row_label=None,
                                           cell_label=None, ))

    return features


processors = {"FinQA": ProcessorFinQA}
