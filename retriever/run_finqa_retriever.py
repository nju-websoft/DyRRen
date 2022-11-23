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
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""
import json
import logging
import math
import os
import string
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed, Trainer, RobertaModel, DataCollatorForTokenClassification,
)
from transformers.trainer_utils import is_main_process

from Config import Config
from models.dyrren_retriever import DyRRenRetriever
from utils.DataCollatorForFinQA import DataCollatorForFinQA
from utils.data_utils import DatasetFinQA, processors, Split
from utils.utils import fin_grained_retrieval_metrics, fin_grained_total_retrieval_metrics

logger = logging.getLogger(__name__)

model_class = {"FinQA": DyRRenRetriever}

dataset_class = {"FinQA": DatasetFinQA}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='./bert-base',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_2_encoder: bool = field(default=False)

    eval_only: bool = field(default=False)

    eval_checkpoints: str = field(default=None)

    pred_mode: str = field(default=None)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_texts_training_retrieval: int = field(default=50,
                                              metadata={"help": "limit texts num when training to fit gpu memory"})
    max_texts_evaluating_retrieval: int = field(default=1000,
                                                metadata={"help": "limit texts num when evaluating to fit gpu memory"})
    max_question_length: int = field(default=48)
    retriever_topn: int = field(default=3)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    if data_args.task_name not in processors:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        # num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    # tokenizer.add_special_tokens({'additional_special_tokens': [Config.NODE_SEP_TOKEN]})
    Config.tokenizer = tokenizer
    Config.punctuation_skiplist = {w: True
                                   for symbol in list(string.punctuation) + [Config.tokenizer.pad_token]
                                   for w in [symbol, Config.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

    def model_init():
        if model_args.eval_only:
            config = AutoConfig.from_pretrained(
                model_args.eval_checkpoints,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
            )
            roberta = AutoModel.from_pretrained(model_args.eval_checkpoints,
                                                from_tf=False,
                                                config=config,
                                                cache_dir=model_args.cache_dir, )
            model = model_class[data_args.task_name].from_pretrained(model_args.eval_checkpoints, roberta=roberta, data_args=data_args)
            return model

        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        roberta = AutoModel.from_pretrained(model_args.model_name_or_path,
                                            from_tf=False,
                                            config=config,
                                            cache_dir=model_args.cache_dir, )
        roberta.resize_token_embeddings(len(tokenizer))
        if model_args.use_2_encoder:
            roberta2 = AutoModel.from_pretrained(model_args.model_name_or_path,
                                                 from_tf=False,
                                                 config=config,
                                                 cache_dir=model_args.cache_dir, )
        else:
            roberta2 = None
        return model_class[data_args.task_name](config=config, roberta=roberta, data_args=data_args, )

    # Get datasets
    train_dataset = (
        dataset_class[data_args.task_name](
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            data_args=data_args,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )

    eval_dataset = (
        dataset_class[data_args.task_name](
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            data_args=data_args,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    test_mode = Split.test
    if model_args.eval_only:
        if model_args.pred_mode == 'train':
            test_mode = Split.train
        elif model_args.pred_mode == 'dev':
            test_mode = Split.dev
        elif model_args.pred_mode == 'test':
            test_mode = Split.test
        elif model_args.pred_mode == 'private_test':
            test_mode = Split.private_test

    test_dataset = (
        dataset_class[data_args.task_name](
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            data_args=data_args,
            overwrite_cache=data_args.overwrite_cache,
            mode=test_mode,
        )
        if training_args.do_predict
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        sim_scores = p.predictions[0].tolist()
        texts_num = p.predictions[1].tolist()
        labels = p.predictions[2].tolist()
        example_ids = p.predictions[3].tolist()

        preds = []
        true_labels = []
        example_indexs = []
        for index, (sim_score, text_num, label, example_id) in enumerate(zip(sim_scores, texts_num, labels, example_ids)):
            if -100 in text_num:
                sim_score = sim_score[:text_num.index(-100)]
                labels = labels[:text_num.index(-100)]
                text_num = text_num[:text_num.index(-100)]
            if len(sim_score) == 0:
                continue
            example_indexs.append(example_id)
            for instance_index in range(len(text_num)):
                text_label = label[sum(text_num[:instance_index]):sum(text_num[:instance_index + 1])]
                text_label = [i if text_label[i] == 1 else -1 for i in range(len(text_label))]
                text_label = list(filter(lambda x: x != -1, text_label))

                # assert len(text_label) > 0
                preds.append(sim_score[sum(text_num[:instance_index]):sum(text_num[:instance_index + 1])])

                true_labels.append(text_label)

        results = fin_grained_total_retrieval_metrics(preds=preds, M=[0] * len(preds), N=[0] * len(preds),
                                                      text_labels=true_labels, cell_labels=[[]] * len(preds),topn=data_args.retriever_topn)
        with open(f'FinQADataset/preds_{test_mode.value}.json', 'w', encoding='utf-8') as output:
            o = [[index, label, pred] for index, pred, label in zip(example_indexs, preds, true_labels)]
            json.dump(o, output, ensure_ascii=False, indent=4)

        return results

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForFinQA(),
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    # Evaluation
    results = {}
    if training_args.do_eval and not model_args.eval_only:
        logger.info("*** Evaluate ***")
        result = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
            results.update(result)

    # Test Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        result = trainer.predict(test_dataset)
        output_eval_file = os.path.join(training_args.output_dir, "pred_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Pred results *****")
            for key, value in result.metrics.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
            results.update(result.metrics)

    return results


if __name__ == "__main__":
    main()
