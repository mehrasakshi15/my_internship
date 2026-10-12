"""
Vocabulary construction for the caption dataset.

Builds a word <-> index mapping from the COCO caption annotations, keeping
only the `vocabulary_size` most frequent words (as in the reference repo).
Special tokens: <pad>=0, <start>=1, <end>=2, <unk>=3
"""
import pickle
import re
from collections import Counter

PAD, START, END, UNK = '<pad>', '<start>', '<end>', '<unk>'


def tokenize(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^a-z0-9' ]", " ", sentence)
    return sentence.split()


class Vocabulary(object):
    def __init__(self, vocabulary_size=5000):
        self.vocabulary_size = vocabulary_size
        self.word2idx = {}
        self.idx2word = {}

    def build(self, list_of_sentences):
        counter = Counter()
        for sent in list_of_sentences:
            counter.update(tokenize(sent))

        # reserve slots for special tokens
        most_common = counter.most_common(self.vocabulary_size - 4)
        words = [PAD, START, END, UNK] + [w for w, _ in most_common]

        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        return self

    def encode(self, sentence, max_len):
        tokens = tokenize(sentence)
        ids = [self.word2idx.get(t, self.word2idx[UNK]) for t in tokens]
        ids = [self.word2idx[START]] + ids + [self.word2idx[END]]
        if len(ids) >= max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [self.word2idx[PAD]] * (max_len - len(ids))
        mask = [1.0 if i != self.word2idx[PAD] else 0.0 for i in ids]
        return ids, mask

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), UNK)
            if w == END:
                break
            if w in (START, PAD):
                continue
            words.append(w)
        return ' '.join(words)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word,
                         'vocabulary_size': self.vocabulary_size}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.vocabulary_size = data['vocabulary_size']
        return self

    def __len__(self):
        return len(self.word2idx)
