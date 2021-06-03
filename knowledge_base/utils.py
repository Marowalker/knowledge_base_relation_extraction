from collections import defaultdict
import random


def make_vocab(infile):
    file = open(infile)
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    raw_vocab = defaultdict()
    id_vocab = defaultdict()
    for idx, line in enumerate(lines):
        if idx != 0:
            pairs = line.split('\t')
            name, name_id = pairs[0], pairs[1]
            raw_vocab[name_id] = name
            id_vocab[name_id] = idx
    return raw_vocab, id_vocab


def process_triples(infile, chem_vocab, dis_vocab, rel_vocab):
    file = open(infile)
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    chems = list(chem_vocab.keys())
    dises = list(dis_vocab.keys())

    heads = []
    tails = []
    rels = []
    heads_neg = []
    tails_neg = []

    for idx, line in enumerate(lines):
        if idx != 0:
            triple = line.split('\t')
            head, tail, rel = triple[0], triple[1], triple[2]
            head_id = chem_vocab[head]
            tail_id = dis_vocab[tail]
            rel_id = rel_vocab[rel]
            threshold = random.random()
            if threshold < 0.5:
                head_neg = random.choice(chems)
                head_neg_id = chem_vocab[head_neg]
                tail_neg_id = tail_id
            else:
                tail_neg = random.choice(dises)
                tail_neg_id = dis_vocab[tail_neg]
                head_neg_id = head_id
            heads.append(head_id)
            tails.append(tail_id)
            rels.append(rel_id)
            heads_neg.append(head_neg_id)
            tails_neg.append(tail_neg_id)

    return heads, tails, rels, heads_neg, tails_neg



