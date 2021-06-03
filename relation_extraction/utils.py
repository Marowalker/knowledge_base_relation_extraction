import codecs
import numpy as np
from collections import defaultdict


class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
        ERROR: Unable to locate file {}.

        FIX: Have you tried running python build_data first?
        This will build vocab file from your train, test and dev sets and
        trim your word vectors.""".format(filename)

        super(MyIOError, self).__init__(message)


def get_trimmed_w2v_vectors(filename):
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def load_vocab(filename):
    try:
        d = dict()
        with codecs.open(filename, encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx + 1  # preserve idx 0 for pad_tok

    except IOError:
        raise MyIOError(filename)
    return d


def max_count_ent(entities):
    temp = []

    for ent in entities:
        if str(-1) in ent:
            entities.remove(ent)
        elif 't' in ent:
            temp.append(ent)
        else:
            if entities.count(ent) == max([entities.count(d) for d in entities]):
                if ent not in temp:
                    temp.append(ent)
    return temp


def load_most_freq_entities():
    file = open('data/cdr/cdr_test.txt')
    lines = file.readlines()
    entity_dict = defaultdict(list)
    most_frequent_ent = defaultdict()
    for line in lines:
        tokens = line.split('\t')
        if len(tokens) == 1:
            title = tokens[0].split('|')
            if 't' in title:
                title_len = len(title[-1])
            else:
                pass
        else:
            # print(title_len)
            if tokens[-2] == 'Chemical':
                if int(tokens[2]) <= title_len - 1:
                    entity_dict[tokens[0]].append(tuple([tokens[-1].strip(), 't']))
                else:
                    entity_dict[tokens[0]].append(tuple([tokens[-1].strip(), 'a']))
            else:
                pass

    for abstract in entity_dict:
        max_ent = max_count_ent(entity_dict[abstract])
        most_frequent_ent[abstract] = max_ent
    return most_frequent_ent


def count_vocab(filename):
    f = open(filename)
    lines = f.readlines()
    return len(lines)

