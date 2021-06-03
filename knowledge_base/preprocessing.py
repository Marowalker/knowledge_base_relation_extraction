from knowledge_base.utils import *
import constants
import os
import pickle

# os.chdir('..')


def make_pickle(infile, outfile):
    chem_name, chem_id = make_vocab(constants.ENTITY_PATH + 'chemical2id.txt')
    dis_name, dis_id = make_vocab(constants.ENTITY_PATH + 'disease2id.txt')
    rel_name, rel_id = make_vocab(constants.ENTITY_PATH + 'relation2id.txt')

    head, tail, rel, head_neg, tail_neg = process_triples(infile, chem_id, dis_id, rel_id)

    sequence_dict = {
        'head': head,
        'tail': tail,
        'rel': rel,
        'head_neg': head_neg,
        'tail_neg': tail_neg
    }
    with open(outfile, 'wb') as f:
        pickle.dump(sequence_dict, f)

    return sequence_dict
