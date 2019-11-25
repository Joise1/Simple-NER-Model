import numpy as np
import pandas as pd


def normalize(x, direction):
    y = x.sum((direction+1) % 2)
    return x.div(y, direction)


class OrgRecognize:

    def __init__(self, input_sentence=''):
        self.q = ['A', 'B', 'C', 'D', 'F', 'G', 'I', 'J', 'K', 'L', 'M', 'P', 'S', 'W', 'X', 'Z']
        self.v = input_sentence
        self.a = self.get_a()
        self.b, self.pi = self.get_b_pi()
        self.patterns = self.get_patterns()

    # get label transform matrix
    @staticmethod
    def get_a():
        label_transform_count_matrix = pd.read_csv('./data/nt.tr.txt', index_col=0)
        return normalize(label_transform_count_matrix, 0)

    # get label to word transform matrix and initial distribution of label
    def get_b_pi(self):
        with open('./data/nt.txt', 'r', encoding='utf-8') as f:
            result = {x: {} for x in self.q}
            pi = {x: 0 for x in self.q}
            line = f.readline()
            while line:
                words = line.split()
                true_word = words[0]
                for index, word in enumerate(words):
                    if index % 2 == 0:
                        continue
                    try:
                        result[word].setdefault(true_word, int(words[index + 1]))
                    except Exception:
                        print(word, line)
                line = f.readline()
            all_count = 0
            for label, words in result.items():
                total_count = 0
                for count in words.values():
                    try:
                        total_count += int(count)
                    except Exception:
                        print(count, words)
                pi[label] += total_count
                all_count += total_count
                result[label] = {word: count / total_count for word, count in result[label].items()}
            pi = {x: count/all_count for x, count in pi.items()}
            return result, pi

    # predict, input word series and output label series
    def predict(self):
        # init
        max_t = len(self.v)
        n = len(self.q)
        delta = pd.DataFrame(index=range(1, max_t+1), columns=self.q, dtype=np.float)
        fai = pd.DataFrame(index=range(1, max_t+1), columns=self.q)
        for label in self.q:
            delta.loc[1, label] = self.pi[label] * self.b[label].setdefault(self.v[0], 0)
        fai[0:1] = pd.DataFrame(np.array([0]*len(self.q)))
        # recursive
        for t in range(2, max_t+1):
            temp = pd.DataFrame(index=self.q, columns=self.q, dtype=float)
            for j in range(0, n):
                temp[self.q[j]] = 1000000*delta.loc[t-1, self.q[j]]*self.a.loc[self.q[j], :]  # *1000000 to avoid overflow
            fai.loc[t, :] = temp.idxmax(1)
            temp = temp.max(1)
            for j in range(0, n):
                delta.loc[t, self.q[j]] = temp[self.q[j]]*self.b[self.q[j]].setdefault(self.v[t-1], 0)
        # stop
        label_t = delta.loc[max_t].idxmax()
        labels = [None]*max_t
        labels[max_t-1] = label_t
        for t in range(max_t-2, -1, -1):
            labels[t] = fai.loc[t+2, labels[t+1]]
        return labels

    # get patterns
    def get_patterns(self):
        result = []
        with open("./data/nt.pattern.txt", "r") as file:
            datas = file.readlines()
            for line in datas:
                result.append(line.strip())
        return result

    # get org
    def get_org(self):
        org_indices = []
        orgs = []
        tag_sequence_str = "".join(self.predict())
        for pattern in self.patterns:
            if pattern in tag_sequence_str:
                start_index = (tag_sequence_str.index(pattern))
                end_index = start_index + len(pattern)
                org_indices.append([start_index, end_index])
        if len(org_indices) != 0:
            for start, end in org_indices:
                orgs.append("".join(self.v[start:end]))
        return orgs


if __name__ == '__main__':
    input_vectors = ['北大', '博物馆', '在', '北京大学', '北侧']  # CDXKB
    ner = OrgRecognize(input_vectors)
    print(ner.predict())
    org = ner.get_org()
    if org:
        print(org)
    else:
        print('There is no organization.')
