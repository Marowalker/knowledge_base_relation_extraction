import constants
from data_utils import *
from knowledge_base.preprocessing import make_pickle
import pickle
from knowledge_base.transE import TransEModel
from relation_extraction.utils import load_vocab, get_trimmed_w2v_vectors
from relation_extraction.dataset import Dataset
from relation_extraction.re_model import REModel
import tensorflow as tf
from evaluate.bc5 import evaluate_bc5


def main_knowledge_base():
    if constants.IS_REBUILD == 1:
        # make_chemicals()
        # make_diseases()
        # make_relations()
        # make_triples()
        # get_train_files()
        train_dict = make_pickle(constants.ENTITY_PATH + 'train2id.txt', constants.PICKLE + 'train_triple_data.pkl')
        val_dict = make_pickle(constants.ENTITY_PATH + 'valid2id.txt', constants.PICKLE + 'val_triple_data.pkl')
        test_dict = make_pickle(constants.ENTITY_PATH + 'test2id.txt', constants.PICKLE + 'test_triple_data.pkl')
    else:
        with open(constants.PICKLE + 'train_triple_data.pkl', 'rb') as f:
            train_dict = pickle.load(f)
            f.close()
        with open(constants.PICKLE + 'val_triple_data.pkl', 'rb') as f:
            val_dict = pickle.load(f)
            f.close()
        with open(constants.PICKLE + 'test_triple_data.pkl', 'rb') as f:
            test_dict = pickle.load(f)

    transe = TransEModel(model_path=constants.TRAINED_MODELS, batch_size=256, epochs=constants.EPOCHS)
    transe.build()
    transe.load_data(train_dict, val_dict)
    transe.train()
    transe.save_model()


def main_re():
    if constants.IS_REBUILD == 1:
        print('Build data')
        # Load vocabularies
        vocab_words = load_vocab(constants.ALL_WORDS)
        vocab_poses = load_vocab(constants.ALL_POSES)
        vocab_synsets = load_vocab(constants.ALL_SYNSETS)
        vocab_depends = load_vocab(constants.ALL_DEPENDS)

        # Create Dataset objects and dump into files
        train = Dataset(constants.SDP + 'sdp_data_with_titles.train.txt',
                        vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                        vocab_depends=vocab_depends)
        pickle.dump(train, open(constants.PICKLE + 'sdp_train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
        dev = Dataset(constants.SDP + 'sdp_data_with_titles.dev.txt',
                      vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                      vocab_depends=vocab_depends)
        pickle.dump(dev, open(constants.PICKLE + 'sdp_dev.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
        test = Dataset(constants.SDP + 'sdp_data_with_titles.test.txt',
                       vocab_words=vocab_words, vocab_poses=vocab_poses, vocab_synset=vocab_synsets,
                       vocab_depends=vocab_depends)
        pickle.dump(test, open(constants.PICKLE + 'sdp_test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        print('Load data')
        train = pickle.load(open(constants.PICKLE + 'sdp_train.pickle', 'rb'))
        dev = pickle.load(open(constants.PICKLE + 'sdp_dev.pickle', 'rb'))
        test = pickle.load(open(constants.PICKLE + 'sdp_test.pickle', 'rb'))

    # Train, Validation Split
    validation = Dataset('', '', process_data=False)
    train_ratio = 0.85
    n_sample = int(len(dev.words) * (2 * train_ratio - 1))
    props = ['words', 'siblings', 'positions_1', 'positions_2', 'labels', 'poses', 'synsets', 'relations', 'directions',
             'identities']
    for prop in props:
        train.__dict__[prop].extend(dev.__dict__[prop][:n_sample])
        validation.__dict__[prop] = dev.__dict__[prop][n_sample:]

    print("Train shape: ", len(train.words))
    print("Test shape: ", len(test.words))
    print("Validation shape: ", len(validation.words))

    # Get word embeddings
    embeddings = get_trimmed_w2v_vectors(constants.TRIMMED_W2V)

    with tf.device('/device:GPU:0'):

        model_re = REModel(constants.TRAINED_MODELS + 're/', embeddings, 256)
        model_re.build(train, validation, test)
        # print(model_re.model.trainable_variables)
        model_re.train(early_stopping=constants.EARLY_STOPPING, patience=constants.PATIENCE)

        # Test on abstract
        answer = {}
        identities = test.identities
        y_pred = model_re.predict()

        # print(identities)
        for i in range(len(y_pred)):
            if y_pred[i] == 0:
                if identities[i][0] not in answer:
                    answer[identities[i][0]] = []

                if identities[i][1] not in answer[identities[i][0]]:
                    answer[identities[i][0]].append(identities[i][1])

        print(
            'result: abstract: ', evaluate_bc5(answer)
        )


if __name__ == '__main__':
    # main_knowledge_base()
    main_re()
