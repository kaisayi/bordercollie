#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/11/9 10:08
"""

from os import path
import codecs
import numpy as np
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def load_data(filename):
    words2id = {}
    label2id = {}
    with codecs.open(filename, encoding="utf8") as fr:
        for line in fr:
            fields = line.strip().split(" ")
            if len(fields) == 2:
                if fields[0] not in words2id:
                    words2id[fields[0]] = len(words2id) + 1
                if fields[1] not in label2id:
                    label2id[fields[1]] = len(label2id) + 1
    words2id["UNK"] = len(words2id) + 1
    return words2id, label2id


class NerDataGenerator(Sequence):
    def __init__(self, data, labels, vocab, max_length, batch_size=32, shuffle=True):
        if not path.exists(data):
            raise FileExistsError("can not find file : %s" % data)
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.data = self._read_data(data)
        self.indexes = np.arange(len(self.data[0]))
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _read_data(self, path):
        sentence_list = []
        label_list = []
        words = []
        labels = []
        with codecs.open(path, encoding="utf8") as fr:
            for line in fr:
                content = line.strip()
                fields = content.split(" ")
                if len(fields) == 2:
                    words.append(self.vocab.get(fields[0], self.vocab["UNK"]))
                    labels.append(self.labels[fields[1]])
                else:
                    if (len(content) == 0) and (len(words) > 0):
                        label_list.append(labels)
                        sentence_list.append(words)
                    words = []
                    labels = []
        x = pad_sequences(sentence_list, self.max_length, padding="post", value=0.)
        y = pad_sequences(label_list, self.max_length, padding="post", value=0.)
        # One-Hot encode
        y = np.array([to_categorical(_y, num_classes=len(self.labels) + 1) for _y in y])
        return x, y

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        _x = self.data[0][indexes]
        _y = self.data[1][indexes]
        return np.array(_x), np.array(_y)

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == '__main__':
    val_data = "../../data/ner/test_data"
    word2id, lable2id = load_data(val_data)
    ng = NerDataGenerator(val_data, lable2id, word2id, 40, batch_size=4)
    xi, yi = ng[0]
    print(xi.shape)
    print(yi.shape)
