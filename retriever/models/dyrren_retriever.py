import logging
import torch
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from Config import Config

logger = logging.getLogger(__name__)


class DyRRenRetriever(RobertaPreTrainedModel):
    def __init__(self, config, roberta: RobertaModel, data_args, roberta2: RobertaModel = None):
        super().__init__(config)
        self.roberta = roberta
        self.roberta2 = roberta2

        if self.roberta2 is not None:
            logging.info(f'use 2 encoder!')
        else:
            logging.info(f'use single encoder!')

        self.hidden_size = config.hidden_size
        self.max_seq_length = data_args.max_seq_length

        self.similarity_metric = 'cosine'

        # self.init_weights()

    def get_table_cell_representation(self, table_representation, table_cell_intervals):
        """
        :param table_representation: Batch x seq length x hidden size
        :param table_cell_intervals: Batch x Batch max cell num x seq length
        :return: Batch x Batch max cell num x hidden size
        """
        raw_cells_representations = torch.bmm(table_cell_intervals, table_representation)
        return raw_cells_representations

    def score(self, Q, D):
        # this method is from ColBERT
        # https://github.com/stanford-futuredata/ColBERT/blob/6493193b98d95595f15cfc375fed2f0b24df4f83/colbert/modeling/colbert.py#L59
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1)).max(-1).values.sum(-1)

    def late_interaction_similarity(self, document_representations, query_representations):
        assert document_representations.size(0) == query_representations.size(0) \
               and document_representations.size(-1) == query_representations.size(-1) == self.hidden_size, \
            f'document: {document_representations.size()}, query: {query_representations.size()}'
        return self.score(query_representations, document_representations, )

    def forward(self, input_ids, attention_mask, token_type_ids, query_input_ids,
                query_attention_mask, query_token_type_ids, text_num, labels, document_mask, example_id):

        document_representations = self.roberta(input_ids,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids)['last_hidden_state']
        if self.roberta2 is None:
            query_representations_raw = self.roberta(query_input_ids,
                                                     attention_mask=query_attention_mask,
                                                     token_type_ids=query_token_type_ids)['last_hidden_state']
        else:
            query_representations_raw = self.roberta2(query_input_ids,
                                                      attention_mask=query_attention_mask,
                                                      token_type_ids=query_token_type_ids)['last_hidden_state']

        query_representations = []
        for instance_index in range(len(text_num)):
            query_representations.append(
                query_representations_raw[instance_index].unsqueeze(0).repeat(text_num[instance_index], 1, 1))

        query_representations = torch.cat(query_representations, dim=0).view(len(document_representations), -1,
                                                                             self.hidden_size)

        mask = document_mask.unsqueeze(2).float()
        document_representations = document_representations * mask

        # NOTE: use vector distance as similarity score
        document_representations = torch.nn.functional.normalize(document_representations, dim=-1)
        query_representations = torch.nn.functional.normalize(query_representations, dim=-1)
        sim_scores = self.late_interaction_similarity(query_representations=query_representations,
                                                      document_representations=document_representations)

        loss = []
        _labels = labels.view(-1)
        assert sim_scores.size(0) == _labels.size(0)
        for i in range(len(text_num)):
            distance_difference = []
            instance_label = _labels[text_num[:i].sum():text_num[:i + 1].sum()]
            instance_sim_scores = sim_scores[text_num[:i].sum():text_num[:i + 1].sum()]

            for label_index in range(len(instance_label)):
                if instance_label[label_index] == 1:
                    for label_index2 in range(len(instance_label)):
                        if instance_label[label_index2] == 1:
                            continue
                        distance_difference.append(instance_sim_scores[label_index2] - instance_sim_scores[label_index])
            loss.append(torch.log(1 + torch.exp(torch.stack(distance_difference, dim=0))).mean())

        loss = torch.stack(loss, dim=0).sum()

        '''
        # NOTE: use MLP to compute similarity score
        sim_scores = self.sim_layer(torch.cat(
            [document_representations.view(-1, self.hidden_size), query_representations.view(-1, self.hidden_size)],
            dim=-1))

        loss = torch.nn.CrossEntropyLoss()(sim_scores, labels.view(-1))
        '''

        '''
        # NOTE: use dot product score to compute similarity score
        document_representations = torch.nn.functional.normalize(document_representations, dim=-1)
        query_representations = torch.nn.functional.normalize(query_representations, dim=-1)
        sim_scores = torch.mm(query_representations, torch.transpose(document_representations, 0, 1))
        sim_scores = torch.nn.functional.log_softmax(sim_scores, dim=-1)

        print(document_representations.size())
        print(query_representations.size())
        print(sim_scores.size())
        print(labels.size())

        loss = self.criterion(sim_scores.view(-1), labels.view(-1))
        '''

        if self.training:
            return loss, sim_scores, text_num, labels, example_id
        else:
            sim_scores = sim_scores.view(1, document_representations.size(0))
            sim_scores = torch.nn.ConstantPad1d((0, 384 - sim_scores.size(1)), 0)(sim_scores)
            labels = torch.nn.ConstantPad1d((0, 384 - labels.size(0)), 0)(labels.unsqueeze(0))
            text_num = text_num.unsqueeze(0)
            example_id = example_id.unsqueeze(0)
            return loss, sim_scores, text_num, labels, example_id