from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch


@dataclass
class DataCollatorForFinQA:
    def __init__(self, pad_token_id, table_pad_token_id, op_list, const_list, max_program_length, max_table_size):
        self.pad_token_id = pad_token_id
        self.table_pad_token_id = table_pad_token_id
        self.use_table_encoder = False if table_pad_token_id is None else True
        self.reserved_token_size = len(op_list) + len(const_list)
        self.max_program_length = max_program_length
        self.max_table_size = max_table_size if self.use_table_encoder else 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [f.__dict__ for f in features]
        batch = {}
        ids = []
        text_input_ids = []
        question_input_ids = []
        table_input_ids = []
        text_attention_mask = []
        question_attention_mask = []
        table_attention_mask = []
        question_mask = []
        text_token_type_id = []
        question_token_type_id = []
        table_token_type_ids = []
        table_cell_spans = []
        table_sizes = []
        text_num = []
        labels = []
        program_ids = []
        option_masks = []
        program_masks = []
        number_indexes = []
        input_masks = []

        _max_text_input_id_len = max([max(list(map(len, features[i]['texts_input_ids']))) for i in range(len(features))])
        _max_question_input_id_len = max([len(features[i]['question_input_ids']) for i in range(len(features))])
        _max_example_numbers_num = max([sum(features[i]['each_part_numbers_num']) for i in range(len(features))])
        _max_example_table_size = max([len(features[i]['table_cell_span']) for i in range(len(features))]) if self.use_table_encoder else 0  # max table size in a batch

        input_id_len = max(_max_text_input_id_len, _max_question_input_id_len)
        table_input_id_len = max([len(features[i]['table_input_ids']) for i in range(len(features))]) if self.use_table_encoder else 0

        def location_2_vector(location, length):
            start_index, end_index = location
            res = np.zeros(length)
            res[start_index:end_index] = 1 / (end_index - start_index)
            return res.tolist()

        def get_option_mask(example_all_text_input_ids, example_number_locations, example_text_number_nums):
            number_index = []
            text_seq_length = len(example_all_text_input_ids[0])
            assert sum(example_text_number_nums) == len(example_number_locations)
            example_text_number_nums = [sum(example_text_number_nums[:i + 1]) for i in range(len(example_text_number_nums) - 1)]
            example_number_locations = np.split(np.array(example_number_locations), example_text_number_nums)
            option_mask = np.zeros((len(example_all_text_input_ids), len(example_all_text_input_ids[0])))
            for text_index, number_location in enumerate(example_number_locations):
                for n in number_location:
                    option_mask[text_index][n[0]] = 1.0  # TODO
                    number_index.append(text_index * text_seq_length + n[0])
            assert len(number_index) == sum(map(len, example_number_locations))
            return option_mask.reshape(-1).tolist(), number_index

        def get_table_option_mask(table_size, table_number_location):
            rows, cols = table_size[0], table_size[1]
            option_mask = [0] * self.max_table_size
            table_number_index = []
            for coor in table_number_location:
                linearized_index = cols * coor[0] + coor[1]
                table_number_index.append(linearized_index)
                try:
                    option_mask[linearized_index] = 1
                except IndexError:
                    print(f'table_index out, table_size {table_size}')
            return option_mask, table_number_index

        for i in range(len(features)):
            def padding_len(i, j):
                return input_id_len - len(features[i]['texts_input_ids'][j])

            def table_padding_len(i):
                return table_input_id_len - len(features[i]['table_input_ids'])

            table_number_num, question_number_num, each_text_numbers_num = features[i]['each_part_numbers_num'][0], \
                                                                           features[i]['each_part_numbers_num'][1], \
                                                                           features[i]['each_part_numbers_num'][2:]
            text_number_location, question_number_location = features[i]['document_number_location'][question_number_num:], features[i]['document_number_location'][
                                                                                                                            :question_number_num]

            ids.append(features[i]['example_id'])

            # Attention: program padding value '0' is the index of operator 'EOF'

            question_padding_len = input_id_len - len(features[i]['question_input_ids'])

            text_num.append(len(features[i]['texts_input_ids']))

            example_text_input_ids = [features[i]['texts_input_ids'][j] + [self.pad_token_id] * padding_len(i, j) for j in range(len(features[i]['texts_input_ids']))]
            text_input_ids += example_text_input_ids

            example_question_input_ids = features[i]['question_input_ids'] + [self.pad_token_id] * question_padding_len
            question_input_ids.append(example_question_input_ids)

            example_table_input_ids = features[i]['table_input_ids'] + [self.table_pad_token_id] * table_padding_len(i) if self.use_table_encoder else []
            table_input_ids.append(example_table_input_ids)

            text_attention_mask += [features[i]['texts_attention_mask'][j] + [0] * padding_len(i, j) for j in
                                    range(len(features[i]['texts_attention_mask']))]
            question_attention_mask.append(features[i]['question_attention_mask'] + [0] * question_padding_len)
            table_attention_mask.append(features[i]['table_attention_mask'] + [0] * table_padding_len(i) if self.use_table_encoder else [])

            text_token_type_id += ([features[i]['texts_token_type_ids'][j] + [0] * padding_len(i, j)
                                    for j in range(len(features[i]['texts_token_type_ids']))]) if features[i]['texts_token_type_ids'] is not None else []
            question_token_type_id.append((features[i]['question_token_type_ids'] + [0] * question_padding_len) if features[i]['question_token_type_ids'] is not None else [])
            table_token_type_ids.append(
                features[i]['table_token_type_ids'] + [[0] * 7 for k in range(table_padding_len(i))] if self.use_table_encoder else [])

            table_cell_spans.append(features[i]['table_cell_span'] + [[-1, -1] for k in range(
                _max_example_table_size - len(features[i]['table_cell_span']))])
            table_sizes.append(features[i]['table_size'])

            document_option_mask, document_number_index = get_option_mask([example_question_input_ids] + example_text_input_ids, features[i]['document_number_location'],
                                                                          features[i]['each_part_numbers_num'][1:])

            table_option_mask, table_number_index = get_table_option_mask(features[i]['table_size'], features[i]['table_number_location']) if self.use_table_encoder else ([], [])

            option_mask = [1] * self.reserved_token_size + table_option_mask + document_option_mask

            option_masks.append(option_mask)

            document_number_index = [n + self.reserved_token_size + self.max_table_size for n in document_number_index]  # max table size is 0 if table encoder not used
            table_number_index = [n + self.reserved_token_size for n in table_number_index] if self.use_table_encoder else []

            number_indexes.append(table_number_index + document_number_index + [-1] * (_max_example_numbers_num - len(table_number_index + document_number_index)))

            _program_ids = features[i]['program_ids'][:self.max_program_length]  # TODO: program ids max length
            # print(_program_ids)
            for j in range(len(_program_ids)):
                if _program_ids[j] == 0:
                    break
                if j % 4 != 0 and j % 4 != 3:
                    if self.use_table_encoder:
                        _program_ids[j] = \
                            document_number_index[_program_ids[j] - self.reserved_token_size - table_number_num] if _program_ids[j] >= self.reserved_token_size + table_number_num \
                                else table_number_index[_program_ids[j] - self.reserved_token_size] if _program_ids[j] >= self.reserved_token_size \
                                else _program_ids[j]
                    else:
                        _program_ids[j] = \
                            document_number_index[_program_ids[j] - self.reserved_token_size] if _program_ids[j] >= self.reserved_token_size else _program_ids[j]

            if 0 not in _program_ids:
                program_masks.append([1] * len(_program_ids))
            else:
                program_masks.append([1] * (_program_ids.index(0) + 1) + [0] * (len(_program_ids) - _program_ids.index(0) - 1))

            program_ids.append(_program_ids)

            for _input_ids in [example_question_input_ids] + example_text_input_ids:
                pad_index = len(_input_ids) if self.pad_token_id not in _input_ids[1:] else _input_ids[1:].index(self.pad_token_id) + 1
                input_masks.append([1] * pad_index + [0] * (len(_input_ids) - pad_index))

            labels.append(features[i]['text_label'])  # this label is for retrieval

            _question_mask = features[i]['question_mask']
            _question_mask = [_q + [0] * (input_id_len - len(_q)) for _q in _question_mask]
            question_mask.append(_question_mask)

        assert max(text_num) == min(text_num)

        new_labels = sum(labels, [])
        assert len(new_labels) == sum(text_num)

        batch['input_ids'] = text_input_ids
        batch['attention_mask'] = text_attention_mask
        batch['token_type_ids'] = text_token_type_id
        batch['query_input_ids'] = question_input_ids
        batch['query_attention_mask'] = question_attention_mask
        batch['query_token_type_ids'] = question_token_type_id
        batch['table_input_ids'] = table_input_ids
        batch['table_attention_mask'] = table_attention_mask
        batch['table_token_type_ids'] = table_token_type_ids
        batch['table_cell_spans'] = table_cell_spans
        batch['table_sizes'] = table_sizes

        batch['text_num'] = text_num
        batch['labels'] = new_labels
        # batch['document_mask'] = [[(x not in Config.punctuation_skiplist) and (x != 0) for x in d] for d in text_input_ids]

        batch['ids'] = ids
        batch['program_ids'] = program_ids
        batch['option_mask'] = option_masks
        batch['program_mask'] = program_masks
        batch['number_index'] = number_indexes
        batch['input_masks'] = input_masks
        batch['question_mask'] = question_mask

        # print(len(each_text_numbers_nums), len(features))
        for key in batch:
            try:
                batch[key] = torch.tensor(batch[key])
            except TypeError:
                print(key)
                print(batch[key])
                exit(0)
            except ValueError:
                print(key)
                print(batch[key])
                exit(0)

        batch['input_ids'] = batch['input_ids'].view(len(ids), -1, input_id_len)
        batch['attention_mask'] = batch['attention_mask'].view(len(ids), -1, input_id_len)
        batch['token_type_ids'] = batch['token_type_ids'].view(len(ids), -1, input_id_len)
        batch['input_masks'] = batch['input_masks'].view(len(ids), -1, input_id_len)

        return batch
