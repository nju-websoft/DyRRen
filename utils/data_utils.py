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
import copy
import json
import logging
import os
import random
import re
import string
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from Config import Config
from utils.utils import table_identify, cell_to_sentence

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    private_test = "private_test"
    dev_gold = "dev_gold"
    test_gold = "test_gold"
    private_test_gold = "private_test_gold"
    dev_and_test = "dev_and_test"


@dataclass(frozen=True)
class InputExampleFinQA:
    example_id: str
    texts: List[str]
    table: List[List[str]]
    question: str
    answer: str
    program: List[Dict[str, str]]
    text_label: List[int]
    question_type: Optional[str]


@dataclass(frozen=True)
class InputFeaturesFinQA:
    example_id: int
    texts_input_ids: List[List[int]]
    question_input_ids: List[List[int]]
    table_input_ids: Optional[List[int]]
    texts_attention_mask: List[List[int]]
    question_attention_mask: List[List[int]]
    table_attention_mask: Optional[List[int]]
    texts_token_type_ids: Optional[List[int]]
    question_token_type_ids: Optional[List[int]]
    table_token_type_ids: Optional[List[List[int]]]
    texts_num: int
    text_label: List[int]
    question_mask: Optional[List[List[int]]]

    table_size: Optional[List[List[int]]]
    table_number_location: Optional[List[List[List[int]]]]
    table_cell_span: Optional[List[List[List[int]]]]  # sorted by row first
    table_cell_mask: Optional[List[List[int]]]

    document_number_location: List[List[int]]  # number locations of question and text, number_location[0] is the location of numbers for question, the rest are for texts
    program_ids: List[int]
    argument_source: List[int]  # number from which text or question, 0-topn-1 is the index of text
    argument_source_mask: List[int]

    each_part_numbers_num: List[
        int]  # number nums of table, question and text, each_part_numbers_num[0] is the num of numbers for table, each_part_numbers_num[1] is the num of numbers for question, the rest are for texts


class ProcessorFinQA:

    def __init__(self, max_texts_training_retrieval, max_texts_evaluating_retrieval, topn_from_retrieval_texts, max_table_size):
        self.max_texts_training_retrieval = max_texts_training_retrieval
        self.max_texts_evaluating_retrieval = max_texts_evaluating_retrieval
        self.topn_from_retrieval_texts = topn_from_retrieval_texts
        self.max_table_size = max_table_size
        self.train_example_ids = []
        self.dev_example_ids = []
        self.test_example_ids = []

        logger.info(f'topn from retrieval texts: {self.topn_from_retrieval_texts}!')

    def get_train_examples(self, tokenizer, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(tokenizer, f'{data_dir}/train.json', "train")

    def get_dev_examples(self, tokenizer, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(tokenizer, f'{data_dir}/dev.json', "dev")

    def get_test_examples(self, tokenizer, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(tokenizer, f'{data_dir}/test.json', "test")

    def _read_json(self, input_dir, data_type):
        datas = json.load(open(input_dir, 'r', encoding='utf-8'))
        retrieval_texts = json.load(open(os.path.join(os.path.dirname(input_dir), f'row_generator_{data_type}.json')))
        retrieval_texts = dict([(d['id'], d) for d in retrieval_texts])
        return datas, retrieval_texts

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
            money = re.compile("^\$ [-+]?[0-9]*\.?[0-9]+")
            res = []

            if re.findall(money, header[-1]):
                for cell in row:
                    text = ("the " + row[0] + " is " + cell + " ; ")
                    res.append(remove_space(text).strip())
            else:
                for head, cell in zip(header, row):
                    text = ("the " + row[0] + " of " + head + " is " + cell + " ; ")  # 注意The大小写
                    res.append(remove_space(text).strip())

            return res

        # cell_sentences = [table_row_to_text(table[0], table[i]) for i in range(len(table))]
        cell_sentences = []
        table, table_class = table_identify(table)
        for i in range(len(table)):
            temp = []
            for j in range(len(table[0])):
                temp.append(cell_to_sentence(table, table_class, i, j))
            cell_sentences.append(temp)
        return cell_sentences

    def _create_examples(self, tokenizer, data_dir, data_type) -> List[InputExampleFinQA]:
        """Creates examples for the training and dev sets."""
        examples = []

        raw_datas, all_retrieval_texts = self._read_json(data_dir, data_type)
        for data_index, data in tqdm(enumerate(raw_datas), total=len(raw_datas), desc=f'Creating {data_type} examples'):
            example_id = data['id']

            if data_type == 'train':
                self.train_example_ids.append(example_id)
            elif data_type == 'dev':
                self.dev_example_ids.append(example_id)
            elif data_type == 'test':
                self.test_example_ids.append(example_id)
            else:
                raise NotImplementedError

            table = data['table']

            while len(table) * len(table[0]) > self.max_table_size:
                table = table[:-1]

            question = data['qa']['question']
            answer = data['qa']['answer']
            program = data['qa']['program']
            gold_texts = list(all_retrieval_texts[example_id]['qa']['gold_inds'].values())

            retrieval_texts = all_retrieval_texts[example_id]['texts'][:self.topn_from_retrieval_texts]
            texts = list(map(lambda x: x['text'], retrieval_texts))

            if data_type == 'train':
                other_texts = list(filter(lambda x: x not in gold_texts, texts))

                while len(gold_texts) >= self.topn_from_retrieval_texts * 2:
                    for i in range(0, len(gold_texts), 2):
                        if i + 1 >= len(gold_texts):
                            continue
                        gold_texts[i] = f'{gold_texts[i]} {gold_texts[i + 1]}'
                        gold_texts[i + 1] = None
                    gold_texts = list(filter(lambda x: x is not None, gold_texts))

                if len(gold_texts) > self.topn_from_retrieval_texts:
                    for i in range(1, len(gold_texts) - self.topn_from_retrieval_texts + 1, 1):
                        gold_texts[-(2 * i - 1)] = f'{gold_texts[-2 * i]} {gold_texts[-(2 * i - 1)]}'
                        gold_texts[-2 * i] = None
                    gold_texts = list(filter(lambda x: x is not None, gold_texts))

                '''
                Note: if want want to only use gold texts for training, use this
                while len(texts) < self.topn_from_retrieval_texts:
                    texts.append('None.')
                '''

                texts = (gold_texts + other_texts)[:self.topn_from_retrieval_texts]
                while len(texts) != self.topn_from_retrieval_texts:
                    if len(other_texts) != 0:
                        texts = (texts + other_texts)[:self.topn_from_retrieval_texts]
                    else:
                        texts.append('None.')
                assert len(texts) == self.topn_from_retrieval_texts

            gold_texts = list(filter(lambda x: x in texts, gold_texts))

            # this may not be True since texts num of some example < topn_from_retrieval_texts.
            # I wish this assert passed but if not, pad retrieval texts to len topn_from_retrieval_texts
            try:
                assert len(texts) == self.topn_from_retrieval_texts
            except AssertionError:
                if len(texts) < self.topn_from_retrieval_texts:
                    texts += ['None.'] * (self.topn_from_retrieval_texts - len(texts))
                    assert len(texts) == self.topn_from_retrieval_texts

            text_label = [1 if texts[i] in gold_texts else 0 for i in range(len(texts))]
            if sum(text_label) == 0 and data_type == 'train':
                print(example_id)
                exit(-1)
            examples.append(InputExampleFinQA(example_id=example_id,
                                              texts=texts,
                                              table=table,
                                              question=question,
                                              answer=answer,
                                              program=program,
                                              text_label=text_label, question_type=None))

        return examples


class ProcessorMultiHiertt:

    def __init__(self, max_texts_training_retrieval, max_texts_evaluating_retrieval, topn_from_retrieval_texts, max_table_size):
        self.max_texts_training_retrieval = max_texts_training_retrieval
        self.max_texts_evaluating_retrieval = max_texts_evaluating_retrieval
        self.topn_from_retrieval_texts = topn_from_retrieval_texts
        self.max_table_size = max_table_size
        self.train_example_ids = []
        self.dev_example_ids = []
        self.test_example_ids = []

        logger.info(f'topn from retrieval texts: {self.topn_from_retrieval_texts}!')

    def get_train_examples(self, tokenizer, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(tokenizer, f'{data_dir}/train.json', "train")

    def get_dev_examples(self, tokenizer, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(tokenizer, f'{data_dir}/dev.json', "dev")

    def get_test_examples(self, tokenizer, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(tokenizer, f'{data_dir}/test.json', "test")

    def _read_json(self, input_dir, data_type):
        datas = json.load(open(input_dir, 'r', encoding='utf-8'))
        retrieval_texts = json.load(open(os.path.join(os.path.dirname(input_dir), f'generator_{data_type}.json')))
        retrieval_texts = dict([(d['id'], d) for d in retrieval_texts])
        return datas, retrieval_texts

    def _text_filter(self, s):
        return s.strip() not in ['', '.']

    def _create_examples(self, tokenizer, data_dir, data_type) -> List[InputExampleFinQA]:
        """Creates examples for the training and dev sets."""
        examples = []

        raw_datas, all_retrieval_texts = self._read_json(data_dir, data_type)
        for data_index, data in tqdm(enumerate(raw_datas), total=len(raw_datas), desc=f'Creating {data_type} examples'):
            example_id = data['uid']

            if data_type == 'train':
                self.train_example_ids.append(example_id)
            elif data_type == 'dev':
                self.dev_example_ids.append(example_id)
            elif data_type == 'test':
                self.test_example_ids.append(example_id)
            else:
                raise NotImplementedError

            question = data['qa']['question']
            answer = data['qa']['answer']
            program = data['qa']['program']
            question_type = data['qa']['question_type']

            if question_type != 'arithmetic' or is_number(program):
                continue

            gold_cell_sentences = [data['table_description'][k] for k in data['qa']['table_evidence']]
            gold_texts = [data['paragraphs'][k] for k in data['qa']['text_evidence']]

            gold_texts = gold_cell_sentences + gold_texts

            cell_sentences = list(data['table_description'].values())

            # mh table handle
            tables = data['tables']
            retrieval_table_idx = int(all_retrieval_texts[example_id]['qa']['retriever_table'][0]) if len(
                all_retrieval_texts[example_id]['qa']['retriever_table']) > 0 else 0  # only top 1
            gold_table_idx = int(all_retrieval_texts[example_id]['qa']['gold_table'][0]) if len(all_retrieval_texts[example_id]['qa']['gold_table']) > 0 else 0  # only top 1


            target_table_idx = gold_table_idx if data_type == 'train' else retrieval_table_idx
            table = pd.read_html(tables[target_table_idx])[0]  # html str to DataFrame

            table_desc = data['table_description']
            all_cell_coors = [[int(co) for co in str(cell_coor_str).split('-')] for cell_coor_str in table_desc.keys()]  # [[0,3,4], [0,3,5] ...]
            target_cell_coors = list(filter(lambda x: x[0] == target_table_idx, all_cell_coors))

            if len(target_cell_coors) > 0:
                max_row = max(list(map(lambda x: x[1], target_cell_coors)))
                max_col = max(list(map(lambda x: x[2], target_cell_coors)))
                min_row = min(list(map(lambda x: x[1], target_cell_coors)))
                min_col = min(list(map(lambda x: x[2], target_cell_coors)))
            else:
                max_row = table.shape[0] - 1
                max_col = table.shape[1] - 1
                min_row = 1
                min_col = 1

            # check for NaN in a row
            while True:
                if min_row == 0:
                    break
                if all(pd.isnull(list(table.iloc[min_row - 1, min_col:]))):
                    min_row -= 1
                    continue
                break

            while True:
                if max_row == table.shape[0] - 1:
                    break
                if all(pd.isnull(list(table.iloc[max_row + 1, min_col:]))):
                    max_row += 1
                    continue
                break

            while min_col > 1:
                table.drop(columns=[1], inplace=True)
                min_col -= 1
                max_col -= 1
                table.columns = range(table.shape[1])
            while max_row + 1 != table.shape[0]:
                table.drop(index=[table.shape[0] - 1], inplace=True)
                table.reset_index(drop=True, inplace=True)
            while max_col != table.shape[1] - 1:
                table.drop(columns=[table.shape[1] - 1], inplace=True)
                table.columns = range(table.shape[1])


            assert min_col == 1
            assert max_row + 1 == table.shape[0]
            assert max_col + 1 == table.shape[1]




            # accumulate headers in a column into one
            col_names = []
            for h_col in range(max_col + 1):
                header_str = ""
                for h_row in range(min_row):
                    header_str += str(table.iloc[h_row, h_col]) + ' '
                col_names.append(header_str.strip())
            table.columns = col_names

            delete_rows = [i for i in range(min_row)]
            table.drop(delete_rows, inplace=True)
            table.reset_index(drop=True, inplace=True)

            # handle numeric value
            for r in range(table.shape[0]):
                for c in range(table.shape[1]):
                    if pd.isnull(table.iloc[r, c]):
                        table.iloc[r, c] = ''
                    elif c > 0:
                        str_value = str(table.iloc[r, c])
                        str_value = str_value.replace('$', '').replace('¥', '').replace(',', '').replace(' %', '%')
                        if is_number(str_value) and str_value[-1] == '%':
                            str_value = str(float(str_value[:-1]) / 100)
                        table.iloc[r, c] = str_value
            table = table.astype(str)

            # convert to list
            list_table_header = col_names
            list_table_values = table.values.tolist()
            assert len(list_table_values[0]) == len(list_table_header)

            table = [list_table_header] + list_table_values

            while len(table) * len(table[0]) > self.max_table_size:
                table = table[:-1]

            retrieval_texts = all_retrieval_texts[example_id]['texts']
            texts = list(map(lambda x: x['text'], retrieval_texts))

            if data_type == 'train':
                other_texts = list(filter(lambda x: x not in gold_texts, texts))

                while len(gold_texts) >= self.topn_from_retrieval_texts * 2:
                    for i in range(0, len(gold_texts), 2):
                        if i + 1 >= len(gold_texts):
                            continue
                        gold_texts[i] = f'{gold_texts[i]} {gold_texts[i + 1]}'
                        gold_texts[i + 1] = None
                    gold_texts = list(filter(lambda x: x is not None, gold_texts))

                if len(gold_texts) > self.topn_from_retrieval_texts:
                    for i in range(1, len(gold_texts) - self.topn_from_retrieval_texts + 1, 1):
                        gold_texts[-(2 * i - 1)] = f'{gold_texts[-2 * i]} {gold_texts[-(2 * i - 1)]}'
                        gold_texts[-2 * i] = None
                    gold_texts = list(filter(lambda x: x is not None, gold_texts))

                '''
                Note: if want want to only use gold texts for training, use this
                while len(texts) < self.topn_from_retrieval_texts:
                    texts.append('None.')
                '''

                texts = (gold_texts + other_texts)
                '''while len(texts) != self.topn_from_retrieval_texts:
                    if len(other_texts) != 0:
                        texts = (texts + other_texts)[:self.topn_from_retrieval_texts]
                    else:
                        texts.append('None.')'''
                # assert len(texts) == self.topn_from_retrieval_texts

            gold_texts = list(filter(lambda x: x in texts, gold_texts))

            # this may not be True since texts num of some example < topn_from_retrieval_texts.
            # I wish this assert passed but if not, pad retrieval texts to len topn_from_retrieval_texts
            '''try:
                assert len(texts) == self.topn_from_retrieval_texts
            except AssertionError:
                if len(texts) < self.topn_from_retrieval_texts:
                    texts += ['None.'] * (self.topn_from_retrieval_texts - len(texts))
                    assert len(texts) == self.topn_from_retrieval_texts'''

            text_label = [1 if texts[i] in gold_texts else 0 for i in range(len(texts))]
            if sum(text_label) == 0 and data_type == 'train':
                print(example_id)
                exit(-1)

            texts = list(map(re_numbert_text, texts))
            question = re_numbert_text(question)

            examples.append(InputExampleFinQA(example_id=example_id,
                                              texts=texts,
                                              table=table,
                                              question=question,
                                              answer=answer,
                                              program=program,
                                              text_label=text_label,
                                              question_type=question_type))

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
            table_tokenizer: PreTrainedTokenizer,
            task: str,
            data_args,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        processor = processors[task](data_args.max_texts_training_retrieval, data_args.max_texts_evaluating_retrieval, data_args.topn_from_retrieval_texts,
                                     data_args.max_table_size)

        max_seq_length = data_args.max_seq_length
        max_question_length = data_args.max_question_length
        max_program_length = data_args.max_program_length

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(max_seq_length),
                table_tokenizer.__class__.__name__ if table_tokenizer is not None else 'NoTab',
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
                    examples = processor.get_dev_examples(tokenizer, data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(tokenizer, data_dir)
                elif mode == Split.train:
                    examples = processor.get_train_examples(tokenizer, data_dir)
                elif mode == Split.dev_and_test:
                    examples = processor.get_dev_examples(tokenizer, data_dir) + processor.get_test_examples(tokenizer, data_dir)
                else:
                    raise NotImplementedError
                logger.info("Training examples: %s", len(examples))
                self.features = convert_examples_to_features_finqa(
                    examples,
                    max_seq_length,
                    max_question_length,
                    tokenizer=tokenizer,
                    table_tokenizer=table_tokenizer,
                    max_program_length=max_program_length,
                    separator_id=tokenizer.encode(' |')[1],
                    filter_id=[tokenizer.encode(' Row')[1]],
                    data_dir=data_dir,
                    mode=mode,
                    train_example_ids=processor.train_example_ids,
                    dev_example_ids=processor.dev_example_ids,
                    test_example_ids=processor.test_example_ids
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeaturesFinQA:
        return self.features[i]


class DatasetMultiHiertt(Dataset):
    features: List[InputFeaturesFinQA]

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            table_tokenizer: PreTrainedTokenizer,
            task: str,
            data_args,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        processor = processors[task](data_args.max_texts_training_retrieval, data_args.max_texts_evaluating_retrieval, data_args.topn_from_retrieval_texts,
                                     data_args.max_table_size)

        max_seq_length = data_args.max_seq_length
        max_question_length = data_args.max_question_length
        max_program_length = data_args.max_program_length

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(max_seq_length),
                table_tokenizer.__class__.__name__ if table_tokenizer is not None else 'NoTab',
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
                    examples = processor.get_dev_examples(tokenizer, data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(tokenizer, data_dir)
                elif mode == Split.train:
                    examples = processor.get_train_examples(tokenizer, data_dir)
                elif mode == Split.dev_and_test:
                    examples = processor.get_dev_examples(tokenizer, data_dir) + processor.get_test_examples(tokenizer, data_dir)
                else:
                    raise NotImplementedError
                logger.info("Training examples: %s", len(examples))
                self.features = convert_examples_to_features_multihiertt(
                    examples,
                    max_seq_length,
                    max_question_length,
                    tokenizer=tokenizer,
                    table_tokenizer=table_tokenizer,
                    max_program_length=max_program_length,
                    separator_id=tokenizer.encode(' |')[1],
                    filter_id=[tokenizer.encode(' Row')[1]],
                    data_dir=data_dir,
                    mode=mode,
                    train_example_ids=processor.train_example_ids,
                    dev_example_ids=processor.dev_example_ids,
                    test_example_ids=processor.test_example_ids, topn_from_retrieval_texts=data_args.topn_from_retrieval_texts
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeaturesFinQA:
        return self.features[i]


def convert_examples_to_features_finqa(
        examples: List[InputExampleFinQA],
        max_length: int,
        max_question_length: int,
        max_program_length: int,
        tokenizer: PreTrainedTokenizer,
        table_tokenizer: PreTrainedTokenizer,
        separator_id: int,
        filter_id: List[int],
        data_dir: str,
        mode=Split.train,
        train_example_ids: List = None,
        dev_example_ids: List = None,
        test_example_ids: List = None,
) -> List[InputFeaturesFinQA]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    features = []

    id_numbers_map = {}

    dsl = json.load(open(os.path.join(data_dir, 'DSL.json')))
    constant_list = dsl['constant_list']
    operator_list = dsl['operator_list']

    assert operator_list[0] == 'EOF'  # the padding value of program ids in DataCollatorForFinQA

    trun_count = 0
    total_count = 0
    t = tqdm(enumerate(examples), total=len(examples), desc=f'Converting {mode} examples to features')
    for data_index, example in t:
        if example.example_id in train_example_ids:
            unique_id = 10000
        elif example.example_id in dev_example_ids:
            unique_id = 20000
        elif example.example_id in test_example_ids:
            unique_id = 30000
        else:
            raise NotImplementedError

        unique_id += data_index

        '''
         ATTENTION: add special token [D] before every text and table, [D]='Document.'
                    add special token [Q] before question, [Q]='Query.'
        '''
        # texts = [f'Document. {text}' for text in example.texts]  # [D] text
        # question = f'Query. {example.question}'

        texts = example.texts
        question = example.question

        extracted_numbers = []

        # table tokenize
        if table_tokenizer is not None:
            # if table encoder is added, table headers is included in table numbers as an argument
            table_headers = None

            tokenized_table = table_tokenize_with_numbers(raw_table=example.table, table_tokenizer=table_tokenizer,
                                                          query=example.question)  # , is_multihiertt=False)
            table_input_ids = tokenized_table['input_ids']
            table_attention_mask = tokenized_table['attention_mask']
            table_token_type_ids = tokenized_table['token_type_ids']
            table_size = tokenized_table['table_size']
            table_numbers = tokenized_table['numbers']
            table_number_location = tokenized_table['number_location']
            table_cell_span = tokenized_table['cell_span']
            table_cell_mask = tokenized_table['cell_mask']
            extracted_numbers += table_numbers

        else:
            # if there is no table encoder, table headers is processed here
            table_headers = [example.table[i][0] for i in range(len(example.table))]
            table_headers = list(filter(lambda x: x.strip() != '', table_headers))

            table_input_ids = []
            table_attention_mask = []
            table_token_type_ids = []
            table_size = []
            table_numbers = []
            table_number_location = []
            table_cell_span = []
            table_cell_mask = []
            extracted_numbers += []

        # tokenized_texts = tokenizer(texts, [question] * len(texts), padding=False, max_length=max_length, truncation=True, return_overflowing_tokens=False, return_length=True)

        tokenized_texts = text_tokenize_with_numbers(texts, text_b=[question] * len(texts), tokenizer=tokenizer, max_length=max_length, headers=table_headers,
                                                     extracted_numbers=extracted_numbers)
        # tokenized_texts = text_tokenize_with_numbers(texts, tokenizer=tokenizer, max_length=max_length, headers=table_headers)
        # tokenized_texts = text_tokenize_with_numbers(texts, tokenizer=tokenizer, max_length=max_length, headers=table_headers)

        texts_input_ids = tokenized_texts['input_ids']
        texts_attention_mask = tokenized_texts['attention_mask']
        texts_token_type_ids = tokenized_texts['token_type_ids'] if 'token_type_ids' in tokenized_texts else None
        texts_question_mask = tokenized_texts['question_mask']

        text_numbers = tokenized_texts['numbers']
        text_number_locations = tokenized_texts['number_location']
        extracted_numbers = tokenized_texts['extracted_numbers']

        if max_question_length == 0:
            # tokenized_question = tokenizer([question], padding=False, max_length=max_length, truncation=True, return_overflowing_tokens=False, return_length=True)
            tokenized_question = text_tokenize_with_numbers([question], tokenizer=tokenizer, max_length=max_length, headers=table_headers, extracted_numbers=extracted_numbers)
            question_input_ids = tokenized_question['input_ids'][0]
            question_attention_mask = tokenized_question['attention_mask'][0]
            question_token_type_ids = tokenized_question['token_type_ids'][0] if 'token_type_ids' in tokenized_question else None
            question_question_mask = tokenized_question['question_mask'][0]
        else:
            tokenized_question = text_tokenize_with_numbers([question], tokenizer=tokenizer, max_length=max_length, headers=table_headers, extracted_numbers=extracted_numbers)
            question_input_ids = tokenized_question['input_ids'][0] + [tokenizer.mask_token_id] * (max_question_length - len(tokenized_question['input_ids'][0]))
            question_attention_mask = [1] * len(question_input_ids)
            question_token_type_ids = [0] * len(question_input_ids) if texts_token_type_ids is not None else None
            question_question_mask = tokenized_question['question_mask'][0]

        question_numbers = tokenized_question['numbers']
        question_number_locations = tokenized_question['number_location']

        each_part_numbers_num = list(map(len, [table_numbers] + question_numbers + text_numbers))  # numbers in table , question, text 0, text 1, ...
        if mode == Split.train:
            program_ids, argument_source, argument_source_mask = program_2_ids(example.program, question_numbers[0], text_numbers, table_numbers, constant_list=constant_list,
                                                                               operator_list=operator_list)[:max_program_length]
            program_ids += [0] * (max_program_length - len(program_ids))
            argument_source += [0] * (max_program_length - len(argument_source))
            argument_source_mask += [0] * (max_program_length - len(argument_source_mask))
        else:
            program_ids = [0] * max_program_length
            argument_source = [0] * max_program_length
            argument_source_mask = [0] * max_program_length
        id_numbers_map[unique_id] = {'example_id': example.example_id, 'numbers': sum([table_numbers] + question_numbers + text_numbers, [])}

        number_location = sum(question_number_locations + text_number_locations, [])
        assert sum(each_part_numbers_num) == len(number_location + table_number_location)

        trun_count += len(list(filter(lambda x: len(x) == max_length, texts_input_ids + [question_input_ids])))  # TODO
        total_count += len(texts_input_ids) + 1
        t.set_description(f'Converting {mode} examples to features. truncation ratio: {trun_count / total_count}')

        features.append(InputFeaturesFinQA(example_id=unique_id,
                                           texts_input_ids=texts_input_ids,
                                           question_input_ids=question_input_ids,
                                           table_input_ids=table_input_ids,
                                           texts_attention_mask=texts_attention_mask,
                                           question_attention_mask=question_attention_mask,
                                           table_attention_mask=table_attention_mask,
                                           texts_token_type_ids=texts_token_type_ids,
                                           question_token_type_ids=question_token_type_ids,
                                           table_token_type_ids=table_token_type_ids,
                                           texts_num=len(texts_input_ids),
                                           text_label=example.text_label,
                                           table_size=table_size,
                                           table_cell_span=table_cell_span,
                                           table_cell_mask=table_cell_mask,
                                           table_number_location=table_number_location,
                                           document_number_location=number_location,
                                           program_ids=program_ids,
                                           argument_source=argument_source,
                                           argument_source_mask=argument_source_mask,
                                           each_part_numbers_num=each_part_numbers_num,
                                           question_mask=texts_question_mask + [question_question_mask]))

    if not os.path.exists(os.path.join(data_dir, os.environ['RUN_NAME'].split('@')[0])):
        os.mkdir(os.path.join(data_dir, os.environ['RUN_NAME'].split('@')[0]))
    with open(os.path.join(data_dir, os.environ['RUN_NAME'].split('@')[0], f'{mode.value}.json'), 'w', encoding='utf-8') as output:
        logging.info(f'Writing {mode.value} example id - numbers into file ' + os.path.join(data_dir, os.environ['RUN_NAME'].split('@')[0], f'{mode.value}.json'))
        json.dump(id_numbers_map, output, ensure_ascii=False, indent=4)

    return features


def convert_examples_to_features_multihiertt(
        examples: List[InputExampleFinQA],
        max_length: int,
        max_question_length: int,
        max_program_length: int,
        tokenizer: PreTrainedTokenizer,
        table_tokenizer: PreTrainedTokenizer,
        separator_id: int,
        filter_id: List[int],
        data_dir: str,
        mode=Split.train,
        train_example_ids: List = None,
        dev_example_ids: List = None,
        test_example_ids: List = None,
        topn_from_retrieval_texts: int = 0,
) -> List[InputFeaturesFinQA]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    features = []

    id_numbers_map = {}

    dsl = json.load(open(os.path.join(data_dir, 'DSL.json')))
    constant_list = dsl['constant_list']
    operator_list = dsl['operator_list']

    assert operator_list[0] == 'EOF'  # the padding value of program ids in DataCollatorForFinQA

    trun_count = 0
    total_count = 0
    t = tqdm(enumerate(examples), total=len(examples), desc=f'Converting {mode} examples to features')
    program_error_count = 0
    for data_index, example in t:
        if example.example_id in train_example_ids:
            unique_id = 10000
        elif example.example_id in dev_example_ids:
            unique_id = 20000
        elif example.example_id in test_example_ids:
            unique_id = 30000
        else:
            raise NotImplementedError

        unique_id += data_index

        '''
         ATTENTION: add special token [D] before every text and table, [D]='Document.'
                    add special token [Q] before question, [Q]='Query.'
        '''
        # texts = [f'Document. {text}' for text in example.texts]  # [D] text
        # question = f'Query. {example.question}'

        texts = example.texts
        question = example.question

        extracted_numbers = []

        table_headers = []

        if table_tokenizer is not None:
            # table tokenize

            tokenized_table = table_tokenize_with_numbers(raw_table=example.table, table_tokenizer=table_tokenizer,
                                                          query=example.question)
            table_input_ids = tokenized_table['input_ids']
            table_attention_mask = tokenized_table['attention_mask']
            table_token_type_ids = tokenized_table['token_type_ids']
            table_size = tokenized_table['table_size']
            table_numbers = tokenized_table['numbers']
            table_number_location = tokenized_table['number_location']
            table_cell_span = tokenized_table['cell_span']
            table_cell_mask = tokenized_table['cell_mask']
            extracted_numbers += table_numbers

        else:
            table_input_ids = []
            table_attention_mask = []
            table_token_type_ids = []
            table_size = []
            table_numbers = []
            table_number_location = []
            table_cell_span = []
            table_cell_mask = []
            extracted_numbers += []

        # tokenized_texts = tokenizer(texts, [question] * len(texts), padding=False, max_length=max_length, truncation=True, return_overflowing_tokens=False, return_length=True)

        tokenized_texts = text_tokenize_with_numbers(texts, text_b=[question] * len(texts), tokenizer=tokenizer, max_length=max_length, headers=table_headers,
                                                     extracted_numbers=extracted_numbers)
        # tokenized_texts = text_tokenize_with_numbers(texts, tokenizer=tokenizer, max_length=max_length, headers=table_headers)
        # tokenized_texts = text_tokenize_with_numbers(texts, tokenizer=tokenizer, max_length=max_length, headers=table_headers)

        texts_input_ids = tokenized_texts['input_ids']
        texts_attention_mask = tokenized_texts['attention_mask']
        texts_token_type_ids = tokenized_texts['token_type_ids'] if 'token_type_ids' in tokenized_texts else None
        texts_question_mask = tokenized_texts['question_mask']

        text_numbers = tokenized_texts['numbers']
        text_number_locations = tokenized_texts['number_location']
        extracted_numbers = tokenized_texts['extracted_numbers']

        filtered_indices = []
        for i in range(len(texts)):
            has_number = False
            for number in extracted_numbers:
                if number in texts[i]:
                    has_number = True
                    break
            filtered_indices.append(has_number)

        def filter_func(ll):
            f = lambda l: [v for v, index_ in zip(l, filtered_indices) if index_][:topn_from_retrieval_texts]
            ll = f(ll)
            while len(ll) < topn_from_retrieval_texts:
                ll = (ll + copy.deepcopy(ll))[:topn_from_retrieval_texts]
            return ll

        texts = filter_func(texts)
        texts_input_ids = filter_func(texts_input_ids)
        texts_attention_mask = filter_func(texts_attention_mask)
        if texts_token_type_ids is not None:
            texts_token_type_ids = filter_func(texts_token_type_ids)
        text_label = filter_func(example.text_label)
        text_numbers = filter_func(text_numbers)
        text_number_locations = filter_func(text_number_locations)
        texts_question_mask = filter_func(texts_question_mask)

        if max_question_length == 0:
            # tokenized_question = tokenizer([question], padding=False, max_length=max_length, truncation=True, return_overflowing_tokens=False, return_length=True)
            tokenized_question = text_tokenize_with_numbers([question], tokenizer=tokenizer, max_length=max_length, headers=table_headers, extracted_numbers=extracted_numbers)
            question_input_ids = tokenized_question['input_ids'][0]
            question_attention_mask = tokenized_question['attention_mask'][0]
            question_token_type_ids = tokenized_question['token_type_ids'][0] if 'token_type_ids' in tokenized_question else None
            question_question_mask = tokenized_question['question_mask'][0]
        else:
            tokenized_question = text_tokenize_with_numbers([question], tokenizer=tokenizer, max_length=max_length, headers=table_headers, extracted_numbers=extracted_numbers)
            question_input_ids = tokenized_question['input_ids'][0] + [tokenizer.mask_token_id] * (max_question_length - len(tokenized_question['input_ids'][0]))
            question_attention_mask = [1] * len(question_input_ids)
            question_token_type_ids = [0] * len(question_input_ids) if texts_token_type_ids is not None else None
            question_question_mask = tokenized_question['question_mask'][0]

        question_numbers = tokenized_question['numbers']
        question_number_locations = tokenized_question['number_location']

        each_part_numbers_num = list(map(len, [table_numbers] + question_numbers + text_numbers))
        if mode == Split.train:
            program_s = program_split(example.program)
            for ps in program_s:
                if is_number(ps) and ps not in sum([table_numbers] + question_numbers + text_numbers, []):
                    # for ii in range(len(table_numbers)):
                    #     if not str(table_numbers[ii]).endswith('%') and table_numbers[ii] != 'nan' and str(int(float(table_numbers[ii]))) == ps:
                    #         table_numbers[ii] = ps
                    for ii in range(len(text_numbers)):
                        for jj in range(len(text_numbers[ii])):
                            if not text_numbers[ii][jj].endswith('%') and text_numbers[ii][jj] != 'Nan' and str(int(float(text_numbers[ii][jj]))) == ps:
                                text_numbers[ii][jj] = ps
            program_ids, argument_source, argument_source_mask, error_occurred = program_2_ids(example.program, question_numbers[0], text_numbers, table_numbers,
                                                                                               constant_list=constant_list,
                                                                                               operator_list=operator_list, is_multihiertt=True)[:max_program_length]
            if error_occurred:
                program_error_count += 1
            program_ids += [0] * (max_program_length - len(program_ids))
            argument_source += [0] * (max_program_length - len(argument_source))
            argument_source_mask += [0] * (max_program_length - len(argument_source_mask))
        else:
            program_ids = [0] * max_program_length
            argument_source = [0] * max_program_length
            argument_source_mask = [0] * max_program_length
        id_numbers_map[unique_id] = {'example_id': example.example_id, 'numbers': sum([table_numbers] + question_numbers + text_numbers, [])}

        number_location = sum(question_number_locations + text_number_locations, [])
        assert sum(each_part_numbers_num) == len(number_location + table_number_location)

        trun_count += len(list(filter(lambda x: len(x) == max_length, texts_input_ids + [question_input_ids])))  # TODO
        total_count += len(texts_input_ids) + 1
        t.set_description(f'Converting {mode} examples to features. truncation ratio: {trun_count / total_count}')

        features.append(InputFeaturesFinQA(example_id=unique_id,
                                           texts_input_ids=texts_input_ids,
                                           question_input_ids=question_input_ids,
                                           table_input_ids=table_input_ids,
                                           texts_attention_mask=texts_attention_mask,
                                           question_attention_mask=question_attention_mask,
                                           table_attention_mask=table_attention_mask,
                                           texts_token_type_ids=texts_token_type_ids,
                                           question_token_type_ids=question_token_type_ids,
                                           table_token_type_ids=table_token_type_ids,
                                           texts_num=len(texts_input_ids),
                                           text_label=text_label,
                                           table_size=table_size,
                                           table_cell_span=table_cell_span,
                                           table_cell_mask=table_cell_mask,
                                           table_number_location=table_number_location,
                                           document_number_location=number_location,
                                           program_ids=program_ids,
                                           argument_source=argument_source,
                                           argument_source_mask=argument_source_mask,
                                           each_part_numbers_num=each_part_numbers_num, question_mask=[question_question_mask] + texts_question_mask))

    if not os.path.exists(os.path.join(data_dir, os.environ['RUN_NAME'].split('@')[0])):
        os.mkdir(os.path.join(data_dir, os.environ['RUN_NAME'].split('@')[0]))
    with open(os.path.join(data_dir, os.environ['RUN_NAME'].split('@')[0], f'{mode.value}.json'), 'w', encoding='utf-8') as output:
        logging.info(f'Writing {mode.value} example id - numbers into file ' + os.path.join(data_dir, os.environ['RUN_NAME'].split('@')[0], f'{mode.value}.json'))
        json.dump(id_numbers_map, output, ensure_ascii=False, indent=4)

    logger.error(f'program error count: {program_error_count}, error ratio: {program_error_count / len(features)}')

    return features


processors = {"FinQA": ProcessorFinQA, "MultiHiertt": ProcessorMultiHiertt}


def re_numbert_text(text):
    # text = text.replace('%$', '$').replace(' %', '%')
    text = text.replace('%$', '$').replace('-$', '-').replace(' %', '%').replace('$', ' $')
    ts = text.split(' ')
    ts = list(filter(lambda x: x.strip() != '', ts))
    # process $/¥
    new_ts = []
    for i in range(len(ts)):
        if ts[i][0] in '$¥' and len(ts[i]) > 1:
            new_ts += [ts[i][0], ts[i][1:]]
        else:
            new_ts.append(ts[i])
    ts = new_ts

    # process punctuation
    new_ts = []
    for i in range(len(ts)):
        if ts[i][-1] in string.punctuation and ts[i][-1] != '%' and len(ts[i]) > 1:
            new_ts += [ts[i][:-1], ts[i][-1]]
        else:
            new_ts.append(ts[i])
    ts = new_ts

    # process number inside ,
    for i in range(len(ts)):
        if ',' in ts[i] and is_number(ts[i].replace(',', '')):
            ts[i] = ts[i].replace(',', '')

    # process -
    new_ts = []
    for i in range(len(ts)):
        if is_number(ts[i]) and ts[i][0] == '-':
            new_ts += [ts[i], '(', ts[i][1:], ')']
        else:
            new_ts.append(ts[i])
    ts = new_ts

    # process %
    new_ts = []
    for i in range(len(ts)):
        if is_number(ts[i]) and ts[i][-1] == '%':
            new_ts += [ts[i], '(', ts[i][:-1], ',', str(float(ts[i][:-1]) / 100), ')']
        else:
            new_ts.append(ts[i])

    return ' '.join(new_ts)


def program_2_ids(p: str, question_numbers: List[str], text_numbers: List[List[str]], table_numbers: List[str], constant_list: List[str], operator_list, is_multihiertt=False):
    error_occurred = False
    p = program_split(p)
    # print(p, numbers)
    assert len(p) % 4 == 0
    s = operator_list + constant_list + table_numbers + question_numbers + sum(text_numbers, [])
    program_ids = []
    argument_source = []
    argument_source_mask = []
    for index_p, _p in enumerate(p):
        # program_ids.append(s.index(_p))
        try:
            program_ids.append(s.index(_p))
        except ValueError:
            if is_multihiertt:
                p[index_p] = 'none'
                program_ids.append(s.index('none'))
                error_occurred = True
            else:
                raise RuntimeError(f'p: {p}\nquestion_numbers: {question_numbers}\ntext_numbers: {text_numbers}')

        argument_source = [0] * len(program_ids)
        argument_source_mask = [0] * len(program_ids)

    if is_multihiertt:
        return program_ids, argument_source, argument_source_mask, error_occurred
    return program_ids, argument_source, argument_source_mask


def program_split(p):
    p = re.split(r'[,(]', p)
    p = list(map(str.strip, p))
    p = sum([[_p] if _p[-1] != ')' else [_p[:-1], _p[-1]] for _p in p], [])
    return p


def is_number(s: str) -> bool:
    if s.strip() == '':
        return False
    if s[-1] == '%':
        s = s[:-1]
    try:
        float(s)
    except ValueError:
        return False
    return True


def table_tokenize_with_numbers(raw_table: List[List[str]], table_tokenizer: PreTrainedTokenizer, query: str):
    result = {}

    pdtable = pd.DataFrame(raw_table[1:], columns=raw_table[0])

    try:
        table_inputs = table_tokenizer(table=pdtable, queries=query, padding='longest', truncation='drop_rows_to_fit', max_length=512,
                                       return_tensors='np')
    except ValueError:
        print(query)
        print(pdtable)
    result['input_ids'] = table_inputs['input_ids'][0].tolist()  # input ids : (1) x max length (1 because only one query)
    result['attention_mask'] = table_inputs['attention_mask'][0].tolist()  # attention mask : (1) x max length
    result['token_type_ids'] = table_inputs['token_type_ids'][0].tolist() if Config.table_model_type == 'tapas' else None  # token_type_ids: (1) x max length x 7 (for tapas)

    table_size = [len(raw_table), len(raw_table[0])]  # row x col
    result['table_size'] = table_size

    numbers = []
    number_location = []
    cell_span = [[-1, -1] for i in range(table_size[1] * table_size[0])]  # (row x col) x 2
    cell_mask = [1] * (table_size[1] * table_size[0])  # row x col

    if Config.table_model_type == 'tapas':
        for idx, each_type in enumerate(table_inputs['token_type_ids'][0]):
            if each_type[0] == 0:
                continue
            row = each_type[2]
            col = each_type[1] - 1
            linearized_idx = row * table_size[1] + col
            if cell_span[linearized_idx][0] == -1:
                cell_span[linearized_idx][0] = idx
                cell_span[linearized_idx][1] = idx
            else:
                cell_span[linearized_idx][1] = idx

        for r in range(table_size[0]):
            for c in range(table_size[1]):
                linearized_idx = r * table_size[1] + c
                if cell_span[linearized_idx][0] == -1 and cell_span[linearized_idx][1] == -1:
                    cell_mask[linearized_idx] = 0
                if r == 0 and raw_table[r][c] != '':
                    numbers.append(raw_table[r][c])
                    number_location.append([r, c])
                    continue
                if c == 0 and raw_table[r][c] != '':
                    numbers.append(raw_table[r][c])
                    number_location.append([r, c])
                    continue
                for single_token in raw_table[r][c].split(' '):
                    if len(single_token) > 0 and is_number(single_token):
                        numbers.append(single_token)
                        number_location.append([r, c])
                        break

    result['numbers'] = numbers
    result['number_location'] = number_location
    result['cell_span'] = cell_span
    result['cell_mask'] = cell_mask
    return result


def get_table_headers_locations_impl(text_input_ids: List[int], tokenizer: PreTrainedTokenizer, headers: List[str]):
    header_locations = []
    contained_headers = []

    def list_contains(list_whole, sublist):
        for i in range(len(list_whole)):
            if list_whole[i] == sublist[0] and list_whole[i:i + len(sublist)] == sublist:
                return i
        return None

    for header in headers:
        # TODO add space but maybe not
        header_tokens = sum([tokenizer.encode(t, add_special_tokens=False) for t in header.split(' ')], [])
        header_start_index = list_contains(text_input_ids, header_tokens)
        if header_start_index is None:
            continue
        contained_headers.append(header)
        header_locations.append([header_start_index, header_start_index + len(header_tokens)])

    return header_locations, contained_headers


def text_tokenize_with_numbers(text: List[str], tokenizer: PreTrainedTokenizer, text_b: List[str] = None, max_length=512, headers: List[str] = None,
                               extracted_numbers=None):
    # Attention: this method can only handle a list of texts insides one example as `extracted_numbers` shares between all texts

    if extracted_numbers is None:
        extracted_numbers = []

    result = {'input_ids': [],
              'attention_mask': [],
              'token_type_ids': [],
              'question_mask': [],
              'numbers': [],
              'number_location': []}

    for text_index, _text in enumerate(text):
        from_front = True
        while True:
            res = text_tokenize_with_numbers_impl(_text, _text_b=text_b[text_index] if text_b is not None else None, tokenizer=tokenizer, max_length=max_length, headers=headers,
                                                  extracted_numbers=extracted_numbers)
            if 'over_length' not in res:
                extracted_numbers += res['numbers']
                break
            current_numbers = res['numbers']
            if text_b is not None:
                _text = _text.split(' ')
                text_b[text_index] = text_b[text_index].split(' ')
                if len(_text) > len(text_b[text_index]):
                    random_del_index = random.randint(0, len(_text) - 1)
                    while _text[random_del_index] in extracted_numbers + current_numbers:
                        random_del_index = random.randint(0, len(_text) - 1)
                    del _text[random_del_index]
                    # _text = _text[1:] if from_front else _text[:-1]
                    # from_front = False if from_front else True
                else:
                    text_b[text_index] = text_b[text_index][:-1]
                _text = ' '.join(_text)
                text_b[text_index] = ' '.join(text_b[text_index])
            else:
                # _text = _text.split(' ')[1:]
                _text = _text.split(' ')
                random_del_index = random.randint(0, len(_text) - 1)
                while _text[random_del_index] in extracted_numbers + current_numbers:
                    random_del_index = random.randint(0, len(_text) - 1)
                del _text[random_del_index]
                # _text = _text[1:] if from_front else _text[: -1]
                # from_front = False if from_front else True
                _text = ' '.join(_text)

        for key in result:
            result[key].append(res[key])

    if Config.model_type == 'roberta':
        del result['token_type_ids']

    result['extracted_numbers'] = extracted_numbers

    return result


def text_tokenize_with_numbers_impl(_text: str, tokenizer: PreTrainedTokenizer, max_length: int, _text_b: str = None, headers: List[str] = None,
                                    extracted_numbers=None):
    if extracted_numbers is None:
        extracted_numbers = []

    _text_input_ids = []
    _token_type_ids = []
    _question_mask = []
    _number_location = []
    _numbers = []

    _text_input_ids.append(tokenizer.cls_token_id)  # add CLS token

    def _process(text, text_input_ids, number_location, numbers):
        for token_index, token in enumerate(text.split(' ')):
            if token == '':
                continue
            _token = token  # if token_index == 0 else ' ' + token  # Attention: add space before every token for roberta tokenizer
            token_input_ids = tokenizer.encode(_token, add_special_tokens=False)

            if is_number(token) and token not in numbers and token not in extracted_numbers:
                number_location.append([len(text_input_ids), len(text_input_ids) + len(token_input_ids)])
                numbers.append(token.strip())
            # elif Config.model_args.duplicate_number_mask and is_number(token):
            #    text_input_ids.append(Config.tokenizer.unk_token_id)
            #    continue

            text_input_ids += token_input_ids
        return text_input_ids, number_location, numbers

    _text_input_ids, _number_location, _numbers = _process(_text, _text_input_ids, _number_location, _numbers)
    _question_mask += [1] * len(_text_input_ids)

    if headers is not None:
        header_locations, contained_headers = get_table_headers_locations_impl(_text_input_ids, tokenizer, headers)
        for header, location in zip(contained_headers, header_locations):
            if header in extracted_numbers or header in _numbers:
                continue
            _number_location.append(location)
            _numbers.append(header)

    if Config.model_type == 'bert':
        _token_type_ids += [0] * len(_text_input_ids)

    if _text_b is not None:
        if Config.model_type == 'bert':
            _text_input_ids.append(tokenizer.sep_token_id)
            _token_type_ids.append(0)

            _text_input_ids += tokenizer.encode(_text_b, add_special_tokens=False)  # Attention: DO NOT detect numbers in text b
            _token_type_ids += [0] * (len(_text_input_ids) - len(_token_type_ids))
        elif Config.model_type == 'roberta':
            _text_input_ids.append(tokenizer.sep_token_id)  # [tokenizer.sep_token_id, tokenizer.sep_token_id]

            for _token in _text_b.split(' '):
                if _token == '':
                    continue
                _text_input_ids += tokenizer.encode(_token, add_special_tokens=False)

        else:
            raise NotImplementedError

    _text_input_ids.append(tokenizer.sep_token_id)
    _question_mask += [0] * (len(_text_input_ids) - len(_question_mask))

    if len(_text_input_ids) > max_length:
        return {'over_length': True, 'numbers': _numbers}

    # print(f'text: {_text}')
    # print(f'text b: {_text_b}')
    # print(f'ids length: {len(_text_input_ids)}')
    # print(f'numbers: {_numbers}')

    _number_location = [[n - 1 for n in nn] for nn in _number_location]

    return {'input_ids': _text_input_ids,
            'attention_mask': [1] * len(_text_input_ids),
            'token_type_ids': _token_type_ids + [1] * (len(_text_input_ids) - len(_token_type_ids)) if len(_token_type_ids) != 0 else None,
            'question_mask': _question_mask,
            'number_location': _number_location,
            'numbers': _numbers}


'''
def get_table_headers_locations_impl(text_input_ids: List[int], tokenizer: PreTrainedTokenizer, headers: List[str]):
    header_locations = []
    contained_headers = []

    def list_contains(list_whole, sublist):
        for i in range(len(list_whole)):
            if list_whole[i] == sublist[0] and list_whole[i:i + len(sublist)] == sublist:
                return i
        return None

    for header in headers:
        # TODO add space but maybe not
        header_tokens = tokenizer.encode(" " + header, add_special_tokens=False)
        header_start_index = list_contains(text_input_ids, header_tokens)
        if header_start_index is None:
            continue
        contained_headers.append(header)
        header_locations.append([header_start_index, header_start_index + len(header_tokens)])

    return header_locations, contained_headers


def text_tokenize_with_numbers(text: List[str], tokenizer: PreTrainedTokenizer, text_b: List[str] = None, max_length=512, headers: List[str] = None,
                               extracted_numbers=None):
    # Attention: this method can only handle a list of texts insides one example as `extracted_numbers` shares between all texts

    if extracted_numbers is None:
        extracted_numbers = []

    result = {'input_ids': [],
              'attention_mask': [],
              'token_type_ids': [],
              'question_mask': [],
              'numbers': [],
              'number_location': []}

    for text_index, _text in enumerate(text):
        from_front = True
        while True:
            res = text_tokenize_with_numbers_impl(_text, _text_b=text_b[text_index] if text_b is not None else None, tokenizer=tokenizer, max_length=max_length, headers=headers,
                                                  extracted_numbers=extracted_numbers)
            if res is not None:
                extracted_numbers += res['numbers']
                break
            if text_b is not None:
                _text = _text.split(' ')
                text_b[text_index] = text_b[text_index].split(' ')
                if len(_text) > len(text_b[text_index]):
                    _text = _text[1:] if from_front else _text[:-1]
                    from_front = False if from_front else True
                else:
                    text_b[text_index] = text_b[text_index][:-1]
                _text = ' '.join(_text)
                text_b[text_index] = ' '.join(text_b[text_index])
            else:
                # _text = _text.split(' ')[1:]
                _text = _text.split(' ')
                _text = _text[1:] if from_front else _text[: -1]
                from_front = False if from_front else True
                _text = ' '.join(_text)

        for key in result:
            result[key].append(res[key])

    if Config.model_type == 'roberta':
        del result['token_type_ids']

    result['extracted_numbers'] = extracted_numbers

    return result


def text_tokenize_with_numbers_impl(_text: str, tokenizer: PreTrainedTokenizer, max_length: int, _text_b: str = None, headers: List[str] = None,
                                    extracted_numbers=None):
    if extracted_numbers is None:
        extracted_numbers = []

    _text_input_ids = []
    _token_type_ids = []
    _question_mask = []
    _number_location = []
    _numbers = []

    _text_input_ids.append(tokenizer.cls_token_id)  # add CLS token

    def _process(text, text_input_ids, number_location, numbers):
        for token_index, token in enumerate(text.split(' ')):
            if token == '':
                continue
            _token = token if token_index == 0 else ' ' + token  # Attention: add space before every token for roberta tokenizer
            token_input_ids = tokenizer.encode(_token, add_special_tokens=False)

            if is_number(token) and token not in numbers and token not in extracted_numbers:
                number_location.append([len(text_input_ids), len(text_input_ids) + len(token_input_ids)])
                numbers.append(token.strip())
            # elif Config.model_args.duplicate_number_mask and is_number(token):
            #    text_input_ids.append(Config.tokenizer.unk_token_id)
            #    continue

            text_input_ids += token_input_ids
        return text_input_ids, number_location, numbers

    _text_input_ids, _number_location, _numbers = _process(_text, _text_input_ids, _number_location, _numbers)
    _question_mask += [1] * len(_text_input_ids)

    if headers is not None:
        header_locations, contained_headers = get_table_headers_locations_impl(_text_input_ids, tokenizer, headers)
        for header, location in zip(contained_headers, header_locations):
            if header in extracted_numbers or header in _numbers:
                continue
            _number_location.append(location)
            _numbers.append(header)

    if Config.model_type == 'bert':
        _token_type_ids += [0] * len(_text_input_ids)

    if _text_b is not None:
        if Config.model_type == 'bert':
            _text_input_ids.append(tokenizer.sep_token_id)
            _token_type_ids.append(0)
        elif Config.model_type == 'roberta':
            _text_input_ids += [tokenizer.sep_token_id, tokenizer.sep_token_id]
        else:
            raise NotImplementedError

        _text_input_ids += tokenizer.encode(_text_b, add_special_tokens=False)  # Attention: DO NOT detect numbers in text b
        if Config.model_type == 'bert':
            _token_type_ids += [0] * (len(_text_input_ids) - len(_token_type_ids))

    _text_input_ids.append(tokenizer.sep_token_id)
    _question_mask += [0] * (len(_text_input_ids) - len(_question_mask))

    if len(_text_input_ids) > max_length:
        return None

    # print(f'text: {_text}')
    # print(f'text b: {_text_b}')
    # print(f'ids length: {len(_text_input_ids)}')
    # print(f'numbers: {_numbers}')
    return {'input_ids': _text_input_ids,
            'attention_mask': [1] * len(_text_input_ids),
            'token_type_ids': _token_type_ids + [1] * (len(_text_input_ids) - len(_token_type_ids)) if len(_token_type_ids) != 0 else None,
            'question_mask': _question_mask,
            'number_location': _number_location,
            'numbers': _numbers}
'''
