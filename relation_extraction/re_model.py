import tensorflow as tf
import constants
from data_utils import Timer, Log
from relation_extraction.utils import count_vocab
import numpy as np
from relation_extraction.dataset import pad_sequences
import os
from sklearn.utils import shuffle
from sklearn.metrics import f1_score


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, embedding, dropout):
        super(CustomLayer, self).__init__()
        # self.embeddings = embeddings
        self.dropout = dropout
        self.embedding = embedding
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.regularizer = tf.keras.regularizers.l2(1e-4)

    def build(self, input_shape):
        # self.w = self.add_weight(name='weights', shape=[self.embedding.shape[-1], self.embedding.shape[-1]],
        #                           dtype=tf.float32, initializer=self.initializer, regularizer=self.regularizer,
        #                          trainable=True)
        self.w = self.embedding

    def call(self, inputs, **kwargs):
        # weighted_embedding = tf.linalg.matmul(self.embedding, self.w)
        lookup = tf.nn.embedding_lookup(params=self.w, ids=inputs)
        return tf.nn.dropout(lookup, self.dropout)


class REModel:
    def __init__(self, model_path, embeddings, batch_size):
        self.trained_models = constants.TRAINED_MODELS
        self.model_path = model_path
        if not os.path.exists(self.trained_models):
            os.makedirs(self.trained_models)
        self.embeddings = embeddings
        self.batch_size = batch_size

        self.max_length = constants.MAX_LENGTH
        # Num of dependency relations
        self.num_of_depend = count_vocab(constants.ALL_DEPENDS)
        # Num of pos tags
        self.num_of_pos = count_vocab(constants.ALL_POSES)
        # num of hypernyms
        self.num_of_synset = count_vocab(constants.ALL_SYNSETS)
        # num_of_siblings
        self.num_of_siblings = count_vocab(constants.ALL_WORDS)
        self.num_of_class = len(constants.ALL_LABELS)
        self.trained_models = constants.TRAINED_MODELS
        self.initializer = tf.keras.initializers.GlorotNormal()

    def _add_inputs(self):
        # Indexes of first channel (word + dependency relations)
        self.word_ids = tf.keras.Input(name='word_ids', shape=(None,), dtype='int32')
        # Indexes of channel (sibling + dependency relations)
        self.sibling_ids = tf.keras.Input(name='sibling_ids', shape=(None, ), dtype='int32')
        # Indexes of third channel (position + dependency relations)
        self.positions_1 = tf.keras.Input(name='positions_1', shape=(None,), dtype='int32')
        # Indexes of third channel (position + dependency relations)
        self.positions_2 = tf.keras.Input(name='positions_2', shape=(None,), dtype='int32')
        # Indexes of second channel (pos tags + dependency relations)
        self.pos_ids = tf.keras.Input(name='pos_ids', shape=(None,), dtype='int32')
        # Indexes of fourth channel (synset + dependency relations)
        self.synset_ids = tf.keras.Input(name='synset_ids', shape=(None,), dtype='int32')

        self.relations = tf.keras.Input(name='relations', shape=(None,), dtype='int32')

    def _add_embedding_ops(self):
        # # Create dummy embedding vector for index 0 (for padding)
        dummy_eb = tf.Variable(np.zeros((1, constants.INPUT_W2V_DIM)), name="dummy", dtype=tf.float32, trainable=False)
        # Create dependency relations randomly
        embeddings_re = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, constants.INPUT_W2V_DIM],
                                                     dtype=tf.float32), name="re_lut")
        # create direction vectors randomly
        embedding_dir = tf.Variable(self.initializer(shape=[3, constants.INPUT_W2V_DIM], dtype=tf.float32),
                                    name="dir_lut")
        # Concat dummy vector and relations vectors
        embeddings_re = tf.concat([dummy_eb, embeddings_re], axis=0)
        # Concat relation vectors and direction vectors
        embeddings_re = tf.concat([embeddings_re, embedding_dir], axis=0)

        # Create word embedding tf variable
        embedding_wd = tf.Variable(self.embeddings, name="lut", dtype=tf.float32, trainable=False)
        embedding_wd = tf.concat([dummy_eb, embedding_wd], axis=0)
        embedding_wd = tf.concat([embedding_wd, embeddings_re], axis=0)

        # initialize word embedding layer
        word_emb = CustomLayer(embedding_wd, constants.DROPOUT)
        self.word_embeddings = word_emb(self.word_ids)

        self.sibling_embeddings = word_emb(self.sibling_ids)

        # Create pos tag embeddings randomly
        dummy_emb_2 = tf.Variable(np.zeros((1, 6)), name='dummy_2', dtype=tf.float32)
        embeddings_pos = tf.Variable(self.initializer(shape=[self.num_of_pos + 1, 6], dtype=tf.float32), name='pos_lut'
                                     , trainable=True)
        embeddings_pos = tf.concat([dummy_emb_2, embeddings_pos], axis=0)
        # Create dependency relations randomly
        embeddings_re2 = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, 6], dtype=tf.float32),
                                     name="re_lut2")
        # create direction vectors randomly
        embedding_dir2 = tf.Variable(self.initializer(shape=[3, 6], dtype=tf.float32), name="dir_lut2")
        # Concat dummy vector and relations vectors
        embeddings_re2 = tf.concat([dummy_emb_2, embeddings_re2], axis=0)
        # Concat relation vectors and direction vectors
        embeddings_re2 = tf.concat([embeddings_re2, embedding_dir2], axis=0)
        embeddings_pos = tf.concat([embeddings_pos, embeddings_re2], axis=0)

        # initialize pos embedding layer
        pos_emb = CustomLayer(embeddings_pos, constants.DROPOUT)
        self.pos_embeddings = pos_emb(self.pos_ids)

        # Create synset embeddings randomly
        dummy_emb_4 = tf.Variable(np.zeros((1, 13)), name='dummy_4', dtype=tf.float32)
        embeddings_synset = tf.Variable(self.initializer(shape=[self.num_of_synset + 1, 13], dtype=tf.float32),
                                        name='syn_lut', trainable=True)
        embeddings_synset = tf.concat([dummy_emb_4, embeddings_synset], axis=0)
        # Create dependency relations randomly
        embeddings_re4 = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, 13], dtype=tf.float32),
                                     name="re_lut4")
        # create direction vectors randomly
        embedding_dir4 = tf.Variable(self.initializer(shape=[3, 13], dtype=tf.float32), name="dir_lut4")
        # Concat dummy vector and relations vectors
        embeddings_re4 = tf.concat([dummy_emb_4, embeddings_re4], axis=0)
        # Concat relation vectors and direction vectors
        embeddings_re4 = tf.concat([embeddings_re4, embedding_dir4], axis=0)
        embeddings_synset = tf.concat([embeddings_synset, embeddings_re4], axis=0)

        # initialize synset embedding layer
        synset_emb = CustomLayer(embeddings_synset, constants.DROPOUT)
        self.synset_embeddings = synset_emb(self.synset_ids)

        # Create position embeddings randomly, each vector has length of WORD EMBEDDINGS / 2
        embeddings_position = tf.Variable(self.initializer(shape=[self.max_length * 2, 25], dtype=tf.float32),
                                          name='position_lut', trainable=True)
        dummy_posi_emb = tf.Variable(np.zeros((1, 25)),
                                     dtype=tf.float32)  # constants.INPUT_W2V_DIM // 2)), dtype=tf.float32)
        embeddings_position = tf.concat([dummy_posi_emb, embeddings_position], axis=0)
        dummy_eb3 = tf.Variable(np.zeros((1, 50)), name="dummy3", dtype=tf.float32, trainable=False)

        embeddings_re3 = tf.Variable(self.initializer(shape=[self.num_of_depend + 1, 50], dtype=tf.float32),
                                     name="re_lut3")
        embeddings_re3 = tf.concat([dummy_eb3, embeddings_re3], axis=0)
        embedding_dir3 = tf.Variable(self.initializer(shape=[3, 50], dtype=tf.float32), name="dir3_lut")
        embeddings_re3 = tf.concat([embeddings_re3, embedding_dir3], axis=0)
        # Concat each position vector with half of each dependency relation vector
        embeddings_position1 = tf.concat([embeddings_position, embeddings_re3[:, :25]],
                                         axis=0)  # :int(constants.INPUT_W2V_DIM / 2)]], axis=0)
        embeddings_position2 = tf.concat([embeddings_position, embeddings_re3[:, 25:]],
                                         axis=0)  # int(constants.INPUT_W2V_DIM / 2):]], axis=0)

        # initialize position embeddings
        position_emb_1 = CustomLayer(embeddings_position1, constants.DROPOUT)
        position_emb_2 = CustomLayer(embeddings_position2, constants.DROPOUT)

        self.position_embeddings_1 = position_emb_1(self.positions_1)
        self.position_embeddings_2 = position_emb_2(self.positions_2)

        self.position_embeddings = tf.concat([self.position_embeddings_1, self.position_embeddings_2], axis=-1)

    def _multiple_input_cnn_layers(self):
        # Create 5-channel features
        self.word_embeddings = tf.expand_dims(self.word_embeddings, -1)
        self.sibling_embeddings = tf.expand_dims(self.sibling_embeddings, -1)
        self.pos_embeddings = tf.expand_dims(self.pos_embeddings, -1)
        self.synset_embeddings = tf.expand_dims(self.synset_embeddings, -1)
        self.position_embeddings = tf.expand_dims(self.position_embeddings, -1)

        # create CNN model
        cnn_outputs = []
        for k in constants.CNN_FILTERS:
            filters = constants.CNN_FILTERS[k]
            cnn_output_w = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=constants.INPUT_W2V_DIM,
                strides=1,
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.word_embeddings)

            cnn_output_sb = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=constants.INPUT_W2V_DIM,
                strides=1,
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.sibling_embeddings)

            cnn_output_pos = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=6,
                strides=1,
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.pos_embeddings)

            cnn_output_synset = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=13,
                strides=1,
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.synset_embeddings)

            cnn_output_position = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=50,
                strides=1,
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(self.position_embeddings)

            cnn_output = tf.concat(
                [cnn_output_w, cnn_output_sb, cnn_output_pos, cnn_output_synset, cnn_output_position],
                axis=1)
            # cnn_output = tf.concat([cnn_output_w, cnn_output_pos, cnn_output_synset, cnn_output_position], axis=1)
            cnn_output = tf.reduce_max(input_tensor=cnn_output, axis=1)
            cnn_output = tf.reshape(cnn_output, [-1, filters])
            cnn_outputs.append(cnn_output)

        final_cnn_output = tf.concat(cnn_outputs, axis=-1)
        final_cnn_output = tf.nn.dropout(final_cnn_output, constants.DROPOUT)

        return final_cnn_output

    def _add_model(self):
        final_cnn_output = self._multiple_input_cnn_layers()
        hidden_1 = tf.keras.layers.Dense(
            units=128, name="hidden_1",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(final_cnn_output)
        hidden_2 = tf.keras.layers.Dense(
            units=128, name="hidden_2",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(hidden_1)
        self.outputs = tf.keras.layers.Dense(
            units=self.num_of_class,
            activation=tf.nn.softmax,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(hidden_2)
        self.model = tf.keras.Model(inputs=[self.word_ids, self.sibling_ids, self.pos_ids, self.synset_ids,
                                            self.positions_1, self.positions_2, self.relations], outputs=self.outputs)

    def _add_metrics(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def build(self, train_data, val_data, test_data):
        timer = Timer()
        timer.start("Building model...")

        self._add_inputs()
        self._load_data(train_data, val_data, test_data)
        self._add_embedding_ops()
        self._add_metrics()
        self._add_model()

        timer.stop()

    def _load_data(self, train_data, val_data, test_data):
        timer = Timer()
        timer.start("Loading data into model...")

        self.dataset_train = train_data
        self.dataset_val = val_data
        self.dataset_test = test_data

        print("Number of training examples:", len(self.dataset_train.labels))
        print("Number of validation examples:", len(self.dataset_val.labels))
        print("Number of test examples:", len(self.dataset_test.labels))

        timer.stop()

    def _next_batch(self, data, num_batch):
        start = 0
        idx = 0
        while idx < num_batch:
            # Get BATCH_SIZE samples each batch
            word_ids = data['words'][start:start + self.batch_size]
            sibling_ids = data['siblings'][start:start + self.batch_size]
            positions_1 = data['positions_1'][start:start + self.batch_size]
            positions_2 = data['positions_2'][start:start + self.batch_size]
            pos_ids = data['poses'][start:start + self.batch_size]
            synset_ids = data['synsets'][start:start + self.batch_size]
            relation_ids = data['relations'][start:start + self.batch_size]
            directions = data['directions'][start:start + self.batch_size]
            labels = data['labels'][start:start + self.batch_size]

            # Padding sentences to the length of longest one
            word_ids, _ = pad_sequences(word_ids, pad_tok=0, max_sent_length=self.max_length)
            sibling_ids, _ = pad_sequences(sibling_ids, pad_tok=0, max_sent_length=self.max_length)
            positions_1, _ = pad_sequences(positions_1, pad_tok=0, max_sent_length=self.max_length)
            positions_2, _ = pad_sequences(positions_2, pad_tok=0, max_sent_length=self.max_length)
            pos_ids, _ = pad_sequences(pos_ids, pad_tok=0, max_sent_length=self.max_length)
            synset_ids, _ = pad_sequences(synset_ids, pad_tok=0, max_sent_length=self.max_length)
            relation_ids, _ = pad_sequences(relation_ids, pad_tok=0, max_sent_length=self.max_length)
            directions, _ = pad_sequences(directions, pad_tok=0, max_sent_length=self.max_length)

            # Create index matrix with words and dependency relations between words
            new_relation_ids = self.embeddings.shape[0] + relation_ids + directions
            word_relation_ids = np.zeros((word_ids.shape[0], word_ids.shape[1] + new_relation_ids.shape[1]))
            w_ids, rel_idxs = [], []
            for j in range(word_ids.shape[1] + new_relation_ids.shape[1]):
                if j % 2 == 0:
                    w_ids.append(j)
                else:
                    rel_idxs.append(j)
            word_relation_ids[:, w_ids] = word_ids
            word_relation_ids[:, rel_idxs] = new_relation_ids

            # Create index matrix with pos tags and dependency relations between pos tags
            new_relation_ids = self.num_of_pos + 1 + relation_ids + directions
            pos_relation_ids = np.zeros((pos_ids.shape[0], pos_ids.shape[1] + new_relation_ids.shape[1]))
            pos_relation_ids[:, w_ids] = pos_ids
            pos_relation_ids[:, rel_idxs] = new_relation_ids

            # Create index matrix with synsets and dependency relations between synsets
            new_relation_ids = self.num_of_synset + 1 + relation_ids + directions
            synset_relation_ids = np.zeros((synset_ids.shape[0], synset_ids.shape[1] + new_relation_ids.shape[1]))
            synset_relation_ids[:, w_ids] = synset_ids
            synset_relation_ids[:, rel_idxs] = new_relation_ids

            # Create index matrix with positions and dependency relations between positions
            new_relation_ids = self.max_length + 1 + relation_ids + directions
            positions_1_relation_ids = np.zeros(
                (positions_1.shape[0], positions_1.shape[1] + new_relation_ids.shape[1]))
            positions_1_relation_ids[:, w_ids] = positions_1
            positions_1_relation_ids[:, rel_idxs] = new_relation_ids

            # Create index matrix with positions and dependency relations between positions
            positions_2_relation_ids = np.zeros(
                (positions_2.shape[0], positions_2.shape[1] + new_relation_ids.shape[1]))
            positions_2_relation_ids[:, w_ids] = positions_2
            positions_2_relation_ids[:, rel_idxs] = new_relation_ids

            start += self.batch_size
            idx += 1
            yield positions_1_relation_ids, positions_2_relation_ids, word_relation_ids, sibling_ids, pos_relation_ids, \
                synset_ids, relation_ids, labels

    def train(self, early_stopping=True, patience=10):
        best_f1 = 0
        n_epoch_no_improvement = 0
        num_batch_train = len(self.dataset_train.labels) // self.batch_size + 1
        for e in range(constants.EPOCHS):
            print("\nStart of epoch %d" % (e + 1,))

            words_shuffled, siblings_shuffled, positions_1_shuffle, positions_2_shuffle, poses_shuffled, \
            synset_shuffled, relations_shuffled, directions_shuffled, labels_shuffled = shuffle(
                self.dataset_train.words,
                self.dataset_train.siblings,
                self.dataset_train.positions_1,
                self.dataset_train.positions_2,
                self.dataset_train.poses,
                self.dataset_train.synsets,
                self.dataset_train.relations,
                self.dataset_train.directions,
                self.dataset_train.labels)

            data = {
                'words': words_shuffled,
                'siblings': siblings_shuffled,
                'positions_1': positions_1_shuffle,
                'positions_2': positions_2_shuffle,
                'poses': poses_shuffled,
                'synsets': synset_shuffled,
                'relations': relations_shuffled,
                'directions': directions_shuffled,
                'labels': labels_shuffled,
            }

            for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_train)):
                positions_1, positions_2, word_ids, sibling_ids, pos_ids, synset_ids, relation_ids, labels = batch
                features = [word_ids, sibling_ids, pos_ids, synset_ids, positions_1, positions_2, relation_ids]
                with tf.GradientTape() as tape:
                    logits = self.model(features, training=True)
                    loss_value = self.loss(labels, logits)
                    loss_value += sum(self.model.losses)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                # Log every 200 batches.
                if idx % 10 == 0:
                    Log.log("Iter {}, Loss: {} ".format(idx, loss_value))

            if early_stopping:
                total_f1 = []
                num_batch_val = len(self.dataset_val.labels) // self.batch_size + 1

                data = {
                    'words': self.dataset_val.words,
                    'siblings': self.dataset_val.siblings,
                    'positions_1': self.dataset_val.positions_1,
                    'positions_2': self.dataset_val.positions_2,
                    'poses': self.dataset_val.poses,
                    'synsets': self.dataset_val.synsets,
                    'relations': self.dataset_val.relations,
                    'directions': self.dataset_val.directions,
                    'labels': self.dataset_val.labels
                }

                for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_val)):
                    positions_1, positions_2, word_ids, sibling_ids, pos_ids, synset_ids, relation_ids, labels = batch
                    features = [word_ids, sibling_ids, pos_ids, synset_ids, positions_1, positions_2, relation_ids]
                    val_acc, f1 = self._accuracy(features, labels)
                    total_f1.append(f1)

                val_f1 = np.mean(total_f1)
                print("Current best F1: ", best_f1)
                print("F1 for epoch number {}: {}".format(e + 1, val_f1))
                if val_f1 > best_f1:
                    self.model.save_weights(self.model_path)
                    Log.log('Save the model at epoch {}'.format(e + 1))
                    best_f1 = val_f1
                    n_epoch_no_improvement = 0
                else:
                    n_epoch_no_improvement += 1
                    Log.log("Number of epochs with no improvement: {}".format(n_epoch_no_improvement))
                    if n_epoch_no_improvement >= patience:
                        print("Best F1: {}".format(best_f1))
                        break

        if not early_stopping:
            self.model.save_weights(self.model_path)

    def _accuracy(self, features, labels):

        logits = self.model(features, training=True)
        accuracy = []
        f1 = []
        predict = []
        exclude_label = []
        for logit, label in zip(logits, labels):
            logit = np.argmax(logit)
            label = np.argmax(label)
            exclude_label.append(label)
            predict.append(logit)
            accuracy += [logit == label]

        f1.append(f1_score(predict, exclude_label, average='macro'))
        return accuracy, np.mean(f1)

    def predict(self):
        y_pred = []
        num_batch = len(self.dataset_test.labels) // self.batch_size + 1

        self.model.load_weights(self.model_path)

        data = {
            'words': self.dataset_test.words,
            'siblings': self.dataset_test.siblings,
            'positions_1': self.dataset_test.positions_1,
            'positions_2': self.dataset_test.positions_2,
            'poses': self.dataset_test.poses,
            'synsets': self.dataset_test.synsets,
            'relations': self.dataset_test.relations,
            'directions': self.dataset_test.directions,
            'labels': self.dataset_test.labels
        }

        for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch)):
            positions_1, positions_2, word_ids, sibling_ids, pos_ids, synset_ids, relation_ids, labels = batch
            features = [word_ids, sibling_ids, pos_ids, synset_ids, positions_1, positions_2, relation_ids]
            logits = self.model(features, training=False)

            for logit in logits:
                decode_sequence = np.argmax(logit)
                y_pred.append(decode_sequence)

        return y_pred



