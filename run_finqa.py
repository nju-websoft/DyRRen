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
import os
import string
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import faulthandler

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed, Trainer, BertModel, )
from transformers.trainer_utils import is_main_process

from Config import Config
from models.fine_grained_retrieval_model import FineGrainedRetrieval
from utils.DataCollatorForFinQA import DataCollatorForFinQA
from utils.data_utils import DatasetFinQA, processors, Split, DatasetMultiHiertt
from utils.evaluate import evaluate_result

logger = logging.getLogger(__name__)

model_class = {"FinQA": FineGrainedRetrieval, "MultiHiertt": FineGrainedRetrieval}  # TODO

dataset_class = {"FinQA": DatasetFinQA, "MultiHiertt": DatasetMultiHiertt}

faulthandler.enable()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='/root/roberta-base',  # '/home/1108037/xli/roberta-large',
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

    table_pretrained_model: str = field(default=None)

    load_ckpt: bool = field(default=False)

    train_checkpoints: str = field(default=None)

    eval_only: bool = field(default=False)

    eval_checkpoints: str = field(default=None)

    model_type: str = field(default='bert')

    decoder_layer_num: int = field(default=1)

    dropout_rate: float = field(default=0.1)

    generator_retrieval: bool = field(default=True)


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
    max_texts_training_retrieval: int = field(default=5,
                                              metadata={"help": "limit texts num when training to fit gpu memory"})
    max_texts_evaluating_retrieval: int = field(default=500,
                                                metadata={"help": "take a big value to not limit when evaluating and predicting"})
    max_question_length: int = field(default=0)

    max_step_index: int = field(default=6)

    topn_from_retrieval_texts: int = field(default=10)

    max_table_size: int = field(default=300)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.max_program_length = data_args.max_step_index * 4

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

    dsl = json.load(open(os.path.join(data_args.data_dir, 'DSL.json')))
    constant_list = dsl['constant_list']
    operator_list = dsl['operator_list']

    # Set seed
    set_seed(training_args.seed)

    if data_args.task_name not in processors:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Example: "<class 'transformers.models.bert.tokenization_bert_fast.BertTokenizerFast'>" --> "bert"
    Config.model_type = str(tokenizer.__class__).split('transformers.models.')[1].split('.')[0]

    pad_token_id = tokenizer.pad_token_id if Config.model_type == 'bert' else tokenizer.cls_token_id
    Config.tokenizer = tokenizer
    Config.model_args = model_args
    Config.task = data_args.task_name

    Config.punctuation_skiplist = {w: True
                                   for symbol in list(string.punctuation) + [Config.tokenizer.pad_token]
                                   for w in [symbol, Config.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

    model_args.fp16 = training_args.fp16

    if model_args.table_pretrained_model is not None:
        table_tokenizer = AutoTokenizer.from_pretrained(model_args.table_pretrained_model)
        table_pad_token_id = table_tokenizer.pad_token_id
        Config.table_tokenizer = table_tokenizer
        # Example: "<class 'transformers.models.tapas.tokenization_tapas.TapasTokenizer'>" --> "tapas"
        Config.table_model_type = str(table_tokenizer.__class__).split('transformers.models.')[1].split('.')[0]
    else:
        table_tokenizer = None
        table_pad_token_id = None

    def model_init():
        dsl = json.load(open(os.path.join(data_args.data_dir, 'DSL.json')))
        constant_list = dsl['constant_list']
        operator_list = dsl['operator_list']

        if model_args.table_pretrained_model is not None:
            table_encoder = AutoModel.from_pretrained(model_args.table_pretrained_model)
        else:
            table_encoder = None

        if model_args.load_ckpt:
            config = AutoConfig.from_pretrained(
                model_args.train_checkpoints,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
            )
            model_args.hidden_size = config.hidden_size

            model = torch.load(os.path.join(model_args.train_checkpoints, 'pytorch_model.bin'))

            roberta_state_dict = list(filter(lambda x: x[0].startswith('roberta.'), model))
            roberta_state_dict = list(map(lambda x: x[1].replace('roberta.', ''), roberta_state_dict))
            roberta = BertModel.from_pretrained(model_args.train_checkpoints, state_dict=dict(roberta_state_dict))

            model = model_class[data_args.task_name].from_pretrained(model_args.train_checkpoints, roberta=roberta, data_args=data_args, model_args=model_args,
                                                                     op_list=operator_list, const_list=constant_list, table_encoder=table_encoder)
            return model

        if model_args.eval_only:
            config = AutoConfig.from_pretrained(
                model_args.eval_checkpoints,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
            )
            model_args.hidden_size = config.hidden_size

            model = torch.load(os.path.join(model_args.eval_checkpoints, 'pytorch_model.bin'))

            roberta_state_dict = list(filter(lambda x: x[0].startswith('roberta.'), model))
            roberta_state_dict = list(map(lambda x: x[1].replace('roberta.', ''), roberta_state_dict))
            roberta = BertModel.from_pretrained(model_args.eval_checkpoints, state_dict=dict(roberta_state_dict))

            model = model_class[data_args.task_name].from_pretrained(model_args.eval_checkpoints, roberta=roberta, data_args=data_args, model_args=model_args,
                                                                     op_list=operator_list, const_list=constant_list, table_encoder=table_encoder)
            return model

        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        model_args.hidden_size = config.hidden_size
        roberta = AutoModel.from_pretrained(model_args.model_name_or_path,
                                            from_tf=False,
                                            config=config,
                                            cache_dir=model_args.cache_dir, )

        return model_class[data_args.task_name](config=config, roberta=roberta, data_args=data_args, model_args=model_args, op_list=operator_list, const_list=constant_list,
                                                table_encoder=table_encoder)

    # Get datasets
    train_dataset = (
        dataset_class[data_args.task_name](
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            table_tokenizer=table_tokenizer,
            task=data_args.task_name,
            data_args=data_args,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
    )

    eval_test_dataset = (
        dataset_class[data_args.task_name](
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            table_tokenizer=table_tokenizer,
            task=data_args.task_name,
            data_args=data_args,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev_and_test,
        )
    )

    dev_test_id_numbers = json.load(open(os.path.join(data_args.data_dir, os.environ['RUN_NAME'].split('@')[0], f'dev_and_test.json'), 'r', encoding='utf-8'))
    processor = processors[data_args.task_name](data_args.max_texts_training_retrieval, data_args.max_texts_evaluating_retrieval, data_args.topn_from_retrieval_texts,
                                                data_args.max_table_size)
    dev_examples = processor.get_dev_examples(tokenizer, data_args.data_dir)
    test_examples = processor.get_test_examples(tokenizer, data_args.data_dir)

    def compute_metrics(p: EvalPrediction) -> Dict:
        steps_scores = p.predictions[0].tolist()
        ids = p.predictions[1].tolist()
        number_index = p.predictions[2].tolist()

        # steps_program_ids = np.argmax(steps_scores, axis=-1).tolist()
        dev_programs = []
        test_programs = []

        for _index, (p_id, program_scores) in enumerate(zip(ids, steps_scores)):
            p_id = str(p_id)

            program_ids = []
            for i in range(len(program_scores)):
                if i % 4 == 0:
                    program_ids.append(program_scores[i].index(max(program_scores[i][:len(operator_list)])))
                else:
                    program_ids.append(program_scores[i][len(operator_list):].index(max(program_scores[i][len(operator_list):])) + len(operator_list))

            for i in range(len(program_ids)):
                # if program_ids[i] == 0:
                #    break
                if i % 4 != 0 and program_ids[i] >= len(constant_list) + len(operator_list):
                    program_ids[i] = number_index[_index].index(program_ids[i]) + len(constant_list) + len(operator_list)

            example_id = dev_test_id_numbers[p_id]['example_id']
            numbers = operator_list + constant_list + dev_test_id_numbers[p_id]['numbers']
            for p in program_ids:
                if p >= len(numbers):
                    logging.error(f'program id not found in numbers!')
                    print(example_id)
                    print(p)
                    print(numbers)
                    print(program_ids)
                    print('\n\n')
            program = [numbers[p] for p in program_ids]

            if 'EOF' in program:
                program = program[:program.index('EOF')]
            program_re = []
            for p_index in range(len(program)):
                if p_index % 4 == 0:
                    program_re.append(f'{program[p_index]}(')
                    if program[p_index].startswith('table_') and p_index + 1 < len(program):
                        program[p_index + 2] = 'none'
                elif p_index % 4 == 3:
                    program_re.append(')')
                else:
                    program_re.append(program[p_index])
            if 20000 <= int(p_id) < 30000:
                dev_programs.append({'id': example_id, 'predicted': program_re + ['EOF']})
            elif int(p_id) >= 30000:
                test_programs.append({'id': example_id, 'predicted': program_re + ['EOF']})
            else:
                raise NotImplementedError

        dev_output_path = os.path.join(training_args.output_dir, f'dev_predicted_{str(datetime.now()).replace(" ", "-").replace(":", "-")}.json')
        with open(dev_output_path, 'w', encoding='utf-8') as output:
            json.dump(dev_programs, output, ensure_ascii=False, indent=4)
        test_output_path = os.path.join(training_args.output_dir, f'test_predicted_{str(datetime.now()).replace(" ", "-").replace(":", "-")}.json')
        with open(test_output_path, 'w', encoding='utf-8') as output:
            json.dump(test_programs, output, ensure_ascii=False, indent=4)

        if data_args.task_name == 'MultiHiertt':
            dev_exe_acc, dev_program_acc = evaluate_result(training_args.output_dir,
                                                           dev_examples,
                                                           os.path.abspath(dev_output_path),
                                                           os.path.abspath(os.path.join(data_args.data_dir, 'val_eval.json')))
            this_results = {'dev_exe_acc': dev_exe_acc, 'dev_program_acc': dev_program_acc}
        else:
            try:
                dev_exe_acc, dev_program_acc = evaluate_result(training_args.output_dir,
                                                               dev_examples,
                                                               os.path.abspath(dev_output_path),
                                                               os.path.abspath(os.path.join(data_args.data_dir, f'dev.json')))
            except TypeError:
                dev_exe_acc = dev_program_acc = 0.0

            try:
                test_exe_acc, test_program_acc = evaluate_result(training_args.output_dir,
                                                                 test_examples,
                                                                 os.path.abspath(test_output_path),
                                                                 os.path.abspath(os.path.join(data_args.data_dir, f'test.json')))
                this_results = {'dev_exe_acc': dev_exe_acc, 'dev_program_acc': dev_program_acc, 'test_exe_acc': test_exe_acc, 'test_program_acc': test_program_acc}
            except TypeError:
                test_exe_acc = test_program_acc = 0.0

        os.system('python /home/xli/TableFineGrainedRetrieval/outputs/del_pt.py')

        return this_results

    set_seed(training_args.seed)

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_test_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForFinQA(pad_token_id=pad_token_id, table_pad_token_id=table_pad_token_id, op_list=operator_list, const_list=constant_list,
                                           max_program_length=data_args.max_program_length, max_table_size=data_args.max_table_size),
    )

    # Training
    if training_args.do_train and not model_args.eval_only:
        trainer.train()
        trainer.save_model()

    # Evaluation
    results = {}
    if training_args.do_eval or model_args.eval_only:
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
        result = trainer.predict(eval_test_dataset)
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
