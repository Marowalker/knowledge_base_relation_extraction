import numpy as np
import constants as constant
from nltk.corpus import wordnet as wn


np.random.seed(13)


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, max_sent_length, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))

        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_sent_length)

    return np.array(sequence_padded), sequence_length


class Dataset:
    # added synset for Dataset class
    def __init__(self, data_name, vocab_words=None, vocab_poses=None, vocab_synset=None, vocab_depends=None,
                 process_data=True):
        self.data_name = data_name

        self.words = None
        self.siblings = None
        self.positions_1 = None
        self.positions_2 = None
        self.labels = None
        self.poses = None
        self.synsets = None
        self.relations = None
        self.directions = None
        self.identities = None

        self.vocab_words = vocab_words
        self.vocab_poses = vocab_poses
        self.vocab_synsets = vocab_synset
        self.vocab_depends = vocab_depends

        if process_data:
            self._process_data()
            self._clean_data()

    def _clean_data(self):
        del self.vocab_words
        del self.vocab_poses
        del self.vocab_synsets
        del self.vocab_depends

    def _process_data(self):
        with open(self.data_name, 'r') as f:
            raw_data = f.readlines()
        data_words, data_siblings, data_postitions, data_y, data_pos, data_synsets, data_relations, \
            data_directions, self.identities = self.parse_raw(raw_data)

        words = []
        siblings = []
        positions_1 = []
        positions_2 = []
        labels = []
        poses = []
        synsets = []
        relations = []
        directions = []

        for i in range(len(data_postitions)):
            position_1, position_2 = [], []
            e1 = data_postitions[i][0]
            e2 = data_postitions[i][-1]
            for po in data_postitions[i]:
                position_1.append((po - e1 + constant.MAX_LENGTH) // 5 + 1)
                position_2.append((po - e2 + constant.MAX_LENGTH) // 5 + 1)
            positions_1.append(position_1)
            positions_2.append(position_2)

        for i in range(len(data_words)):
            rs = []
            for r in data_relations[i]:
                rid = self.vocab_depends[r]
                rs += [rid]
            relations.append(rs)

            ds = []
            for d in data_directions[i]:
                did = 1 if d == 'l' else 2
                ds += [did]
            directions.append(ds)

            ws, ps, ss, sbs = [], [], [], []

            for w, sb, p, s in zip(data_words[i], data_siblings[i], data_pos[i], data_synsets[i]):
                if w in self.vocab_words:
                    word_id = self.vocab_words[w]
                else:
                    word_id = self.vocab_words[constant.UNK]
                ws.append(word_id)

                temp = []
                for token_word in sb:
                    if token_word in self.vocab_words:
                        sibling_id = self.vocab_words[token_word]
                    else:
                        sibling_id = self.vocab_words[constant.UNK]
                    sbs.append(sibling_id)
                    # temp.append(sibling_id)
                # sbs.append(temp)

                if p in self.vocab_poses:
                    p_id = self.vocab_poses[p]
                else:
                    p_id = self.vocab_poses['NN']

                ps += [p_id]
                if s in self.vocab_synsets:
                    synset_id = self.vocab_synsets[s]
                else:
                    synset_id = self.vocab_synsets['entity.n.01']
                ss += [synset_id]

            words.append(ws)
            siblings.append(sbs)
            poses.append(ps)
            synsets.append(ss)

            if data_y[i][0] == 'CID':
                lb = [1, 0]
            else:
                lb = [0, 1]
            # lb = constant.ALL_LABELS.index(data_y[i][0])
            labels.append(lb)

        self.words = words
        self.siblings = siblings
        self.positions_1 = positions_1
        self.positions_2 = positions_2
        self.labels = labels
        self.poses = poses
        self.synsets = synsets
        self.relations = relations
        self.directions = directions

    def parse_raw(self, raw_data):
        all_words = []
        all_siblings = []
        all_positions = []
        all_relations = []
        all_directions = []
        all_poses = []
        all_labels = []
        all_synsets = []
        all_identities = []
        pmid = ''
        for line in raw_data:
            l = line.strip().split()
            if len(l) == 1:
                pmid = l[0]
                # print()
                # print(pmid)
            else:
                pair = l[0]
                label = l[1]
                # print(pair, label)
                if label:
                    joint_sdp = ' '.join(l[2:])
                    sdps = joint_sdp.split("-PUNC-")
                    for sdp in sdps:
                        # S xuoi
                        nodes = sdp.split()
                        words = []
                        siblings = []
                        positions = []
                        poses = []
                        synsets = []
                        relations = []
                        directions = []

                        for idx, node in enumerate(nodes):
                            node = node.split('|')

                            if idx % 2 == 0:
                                sibling_word = []
                                for idx, _node in enumerate(node):
                                    word = constant.UNK if _node == '' else _node
                                    if idx == 0:
                                        w, p, s = word.split('\\')
                                        p = 'NN' if p == '' else p
                                        s = str(wn.synsets('a')[0].name()) if s == '' else s
                                        _w, position = w.rsplit('_', 1)
                                        words.append(_w)
                                        positions.append(min(int(position), constant.MAX_LENGTH))
                                        poses.append(p)
                                        synsets.append(s)
                                    else:
                                        sibling_word.append(word)
                                siblings.append(sibling_word)
                            else:
                                dependency = node[0]
                                r = '(' + dependency[3:]
                                d = dependency[1]
                                r = r.split(':', 1)[0] + ')' if ':' in r else r
                                relations.append(r)
                                # relations.append(dependency)
                                directions.append(d)

                        all_words.append(words)
                        all_siblings.append(siblings)
                        all_positions.append(positions)
                        all_relations.append(relations)
                        all_directions.append(directions)
                        all_poses.append(poses)
                        all_synsets.append(synsets)
                        all_labels.append([label])
                        all_identities.append((pmid, pair))
                else:
                    print(l)

        return all_words, all_siblings, all_positions, all_labels, all_poses, all_synsets, all_relations, \
            all_directions, all_identities
