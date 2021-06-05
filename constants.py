import argparse


parser = argparse.ArgumentParser(description='Knowledge-based NLP model for re')
parser.add_argument('-i', help='Job identity', type=int, default=0)
parser.add_argument('-rb', help='Rebuild data', type=int, default=0)
parser.add_argument('-e', help='Number of epochs', type=int, default=80)
parser.add_argument('-p', help='Patience of early stop (0 to ignore)', type=int, default=10)
parser.add_argument('-len', help='Max sentence or document length', type=int, default=100)
parser.add_argument('-d', help='dropout rate', type=int, default=0.5)
parser.add_argument('-config', help='CNN configurations default \'1:128\'', type=str, default='2:32,3:96,4:32,5:64')
parser.add_argument('-sc', help='scoring function for TransE', type=str, default='l2')

opt = parser.parse_args()
print('Running opt: {}'.format(opt))

JOB_IDENTITY = opt.i
IS_REBUILD = opt.rb
EPOCHS = opt.e
EARLY_STOPPING = False if opt.p == 0 else True
MAX_LENGTH = opt.len
DROPOUT = opt.d
PATIENCE = opt.p
SCORE = opt.sc

CNN_FILTERS = {}
if opt.config:
    print('Use model CNN with config', opt.config)
    USE_CNN = True
    CNN_FILTERS = {
        int(k): int(f) for k, f in [i.split(':') for i in opt.config.split(',')]
    }
else:
    raise ValueError('Configure CNN model to start')

UNK = '$UNK$'

ALL_LABELS = ['CID', 'NONE']


columns = {'ChemicalName', 'ChemicalID', 'CasRN', 'DiseaseName', 'DiseaseID', 'DirectEvidence', 'InferenceGeneSymbol',
           'InferenceScore', 'OmimIDs', 'PubMedIDs'}

list_relations = ['inferred-association', 'therapeutic', 'marker/mechanism', 'other']

DATA = 'data/'

CTD = DATA + 'ctd/CTD_chemicals_diseases.csv'

CDR = DATA + 'cdr/cdr_'

PICKLE = DATA + 'pickle/'

cdr_datasets = ['train.txt', 'dev.txt', 'test.txt']

ENTITY_PATH = DATA + 'entity_representation/'

INPUT_W2V_DIM = 200

TRAINED_MODELS = DATA + 'trained_models/'

SDP = DATA + 'sdp/'
W2V_DATA = DATA + 'w2v_model/'

ALL_WORDS = DATA + 'all_words.txt'
ALL_POSES = DATA + 'all_poses.txt'
ALL_SYNSETS = DATA + 'all_hypernyms.txt'
ALL_DEPENDS = DATA + 'all_relations.txt'

TRIMMED_W2V = W2V_DATA + 'biowordvec_nlplab.npz'

