# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import tqdm


def get_corpus(data_path):
    corpus = []
    with open(data_path, 'r') as f:
        line = f.readline()
        sentence = []
        tags = []
        while line:
            if line.strip():  # There is a blank line between sentences.
                char, label = line.strip().split()
                sentence.append(char)
                tags.append(label)
            else:
                corpus.append((sentence, tags))
                sentence = []
                tags = []
            line = f.readline()
    return corpus


def get_word_dict(corpus):
    word2ix = {}
    for sent, tags in corpus:
        for word in sent:
            if word not in word2ix:
                word2ix[word] = len(word2ix)
    return word2ix


def get_embedding(word2ix, dim):
    n = len(word2ix)
    embeddings = nn.Embedding(n, dim)
    torch.manual_seed(1)
    return embeddings


def get_keys(value, dic):
    return [k for k, v in dic.items() if v == value]


def seq2ix(seq, ix):
    idxs = []
    for x in seq:
        if x in ix.keys():
            idxs.append(ix[x])
        elif idxs:
            idxs.append(idxs[-1])
        else:
            idxs.append(0)
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


def ix2seq(seq, ix):
    key_seq = []
    for element in seq:
        key_seq.append(get_keys(element, ix)[0])
    return key_seq


def log_sum_exp(x):
    exp = torch.exp(x)
    return torch.log(torch.sum(exp))


class BiLSTMCrf(nn.Module):

    def __init__(self, input_dim, hidden_dim, word_embeddings, tag_dict, layers_num=1, dropout=0.5):
        super(BiLSTMCrf, self).__init__()
        self.corpus = corpus
        self.tag_dict = tag_dict
        self.tag_size = len(self.tag_dict)
        self.word_embeddings = word_embeddings

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        self.dropout = dropout
        self.batch_size = 1
        self.bi_flag = True
        self.bi_num = 2 if self.bi_flag else 1

        self.lstm_layer = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers_num,
                            dropout=dropout, bidirectional=self.bi_flag)
        self.project_layer = nn.Linear(self.hidden_dim * self.bi_num, self.tag_size)
        self.hidden = self.init_hidden()

        self.transition_matrix = nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        self.transition_matrix.data[:, tag_dict[start_tag]] = -10000
        self.transition_matrix.data[tag_dict[stop_tag], :] = -10000

    def init_hidden(self):
        return (Variable(torch.zeros(self.layers_num * self.bi_num, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.layers_num * self.bi_num, self.batch_size, self.hidden_dim)))

    def decode(self, feats):
        viterbi_var = torch.full([self.tag_size], -10000)
        viterbi_var[self.tag_dict[start_tag]] = 0
        viterbi_pointer = torch.Tensor(feats.size())
        for t, feat in enumerate(feats):
            temp_viterbi = viterbi_var.unsqueeze(0).transpose(0, 1).repeat(1, self.tag_size) + self.transition_matrix
            viterbi_var = temp_viterbi.max(dim=0).values + torch.Tensor(feat)
            viterbi_pointer[t] = temp_viterbi.argmax(dim=0)

        viterbi_var += self.transition_matrix[:, self.tag_dict[stop_tag]].clone()
        score = viterbi_var.max()
        best_tag = viterbi_var.argmax()
        best_tags = [None] * (feats.size()[0]+1)
        best_tags[-1] = best_tag
        for t in range(feats.size()[0] - 1, -1, -1):
            best_tags[t] = viterbi_pointer[t][int(best_tags[t+1].item())]
        best_tags.pop(0)
        return score, best_tags

    def all_path_score(self, feats):
        forward = torch.full([self.tag_size], -10000)
        forward[self.tag_dict[start_tag]] = 0

        for feat in feats:
            emit_score = torch.unsqueeze(feat, 0).transpose(0, 1)
            trans_score = self.transition_matrix.transpose(0, 1)
            temp_forward = forward.repeat(self.tag_size, 1)
            forward = torch.logsumexp(emit_score+trans_score+temp_forward, dim=1)

        # forward += self.transition_matrix[:, self.tag_dict[stop_tag]].clone()
        final_forward = forward + self.transition_matrix[:, self.tag_dict[stop_tag]].clone()
        return log_sum_exp(final_forward)

    def get_score(self, feats, tags):
        score = 0
        new_tags = torch.cat([torch.LongTensor([self.tag_dict[start_tag]]), tags])
        for t, feat in enumerate(feats):
            tag = new_tags[t]
            next_tag = new_tags[t+1]
            score += feat[next_tag] + self.transition_matrix[tag][next_tag]
        score += self.transition_matrix[new_tags[-1]][self.tag_dict[stop_tag]]
        return score

    def lstm(self, sent):
        self.hidden = self.init_hidden()
        sent_embeddings = self.word_embeddings(sent)
        output, self.hidden = self.lstm_layer(sent_embeddings.reshape(len(sent), self.batch_size, -1), self.hidden)

        tag_space = self.project_layer(output.reshape(len(sent), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

    def loss_function(self, sentence, tags):
        feats = self.lstm(sentence)
        all_path_score = self.all_path_score(feats)
        tag_score = self.get_score(feats, tags)
        return all_path_score - tag_score

    def forward(self, sent):
        # bi-lstm
        tag_scores = self.lstm(sent)
        # crf
        score, tag_seq = self.decode(tag_scores)
        return score, tag_seq


def evaluate_sent(true_tag, predict_tag):
    gold_num = 0
    predict_num = 0
    correct_num = 0
    sent_len = len(true_tag)
    start_flag = False
    equal_flag = True

    for i in range(sent_len):
        gold_num = gold_num + 1 if 'B' in true_tag[i] else gold_num
        predict_num = predict_num + 1 if 'B' in predict_tag[i] else predict_num
        if 'B' in true_tag[i]:
            start_flag = True
        if start_flag and true_tag[i] != predict_tag[i]:
            equal_flag = False
        if start_flag and (i < sent_len - 1 and 'I' not in true_tag[i+1]) or i == sent_len - 1:
            start_flag = False
            if equal_flag and (i == sent_len - 1 or 'I' not in true_tag[i + 1]):
                correct_num += 1
            equal_flag = True
    return gold_num, predict_num, correct_num


if __name__ == '__main__':
    # model parameters
    epoch_num = 3
    word_embedding_dim = 300
    hidden_dim = 256
    start_tag = '<START>'
    stop_tag = '<STOP>'
    train_path = './data/train_data'
    test_path = './data/test_data'
    tag_dict = {'B-ORG': 0, 'I-ORG': 1, '0': 2, 'O': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-PER': 5, 'I-PER': 6,
                start_tag: 7, stop_tag: 8}
    # preprocess
    corpus = get_corpus(train_path)
    word_dict = get_word_dict(corpus)
    print('len of word dict', len(word_dict))
    word_embeddings = get_embedding(word_dict, word_embedding_dim)
    # model build
    model = BiLSTMCrf(word_embedding_dim, hidden_dim, word_embeddings, tag_dict)
    # model train
    print('begin model training.')
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epoch_num):
        for sentence, tags in tqdm.tqdm(corpus):
            model.zero_grad()

            sent_in = seq2ix(sentence, word_dict)
            true_tags = seq2ix(tags, tag_dict)

            loss = model.loss_function(sent_in, true_tags)

            loss.backward()
            optimizer.step()
    # model predict
    print('begin model predicting.')
    test_data = get_corpus(test_path)
    gold_num = 0
    predict_num = 0
    correct_num = 0
    for sent, tags in tqdm.tqdm(test_data):
        sent_ix = seq2ix(sent, word_dict)
        with torch.no_grad():
            score, predict_tag = model(sent_ix)
            predict_tag = ix2seq(predict_tag, tag_dict)
            new_gold_num, new_predict_num, new_correct_num = evaluate_sent(tags, predict_tag)
            gold_num += new_gold_num
            predict_num += new_predict_num
            correct_num += new_correct_num
    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f_score = (2*precision*recall) / (precision + recall)
    print(gold_num, predict_num, correct_num)
    print(precision, recall, f_score)