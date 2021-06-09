import tensorflow as tf
from sklearn.utils import shuffle
import constants
import os
import numpy as np
from data_utils import count_vocab, Timer, Log


# tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()


class LookupLayer(tf.keras.layers.Layer):
    def __init__(self, embeddings):
        super(LookupLayer, self).__init__()
        self.embeddings = embeddings

    def build(self, input_shape):
        self.w = self.add_weight('embedding', shape=self.embeddings.shape,
                                 initializer=tf.keras.initializers.GlorotNormal(),
                                 regularizer=tf.keras.regularizers.l2(1e-4), trainable=True)

    def call(self, inputs, **kwargs):
        emb = self.embeddings * self.w
        return tf.nn.embedding_lookup(params=emb, ids=inputs)


class TransEModel:
    def __init__(self, embeddings, model_path, batch_size, epochs, score):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.chem_size = count_vocab(constants.ENTITY_PATH + 'chemical2id.txt')
        self.dis_size = count_vocab(constants.ENTITY_PATH + 'disease2id.txt')
        self.rel_size = count_vocab(constants.ENTITY_PATH + 'relation2id.txt')
        self.epochs = epochs
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        self.dataset_train = None
        self.dataset_val = None
        self.score = score

    def _add_inputs(self):
        self.head = tf.keras.Input(name='head', shape=(None,), dtype=tf.int32)
        self.tail = tf.keras.Input(name='tail', shape=(None,), dtype=tf.int32)
        self.rel = tf.keras.Input(name='rel', shape=(None,), dtype=tf.int32)
        self.head_neg = tf.keras.Input(name='head_neg', shape=(None,), dtype=tf.int32)
        self.tail_neg = tf.keras.Input(name='tail_neg', shape=(None,), dtype=tf.int32)

    def _add_embeddings(self):
        # generate embeddings
        self.chemical_embeddings = tf.Variable(self.embeddings,
                                               dtype=tf.float32, name='chemicals', trainable=False)
        self.chemical_embeddings = tf.nn.l2_normalize(self.chemical_embeddings, axis=1)
        self.disease_embedings = tf.Variable(self.embeddings,
                                             dtype=tf.float32, name='diseases', trainable=False)
        self.disease_embedings = tf.nn.l2_normalize(self.disease_embedings, axis=1)
        self.relation_embeddings = tf.Variable(self.initializer(shape=[self.rel_size + 1, constants.INPUT_W2V_DIM]),
                                               dtype=tf.float32, name='relation', trainable=False)
        self.relation_embeddings = tf.nn.l2_normalize(self.relation_embeddings, axis=1)

        # lookup embedding for scoring function
        chemical_lookup = LookupLayer(self.chemical_embeddings)
        disease_lookup = LookupLayer(self.disease_embedings)
        relation_lookup = LookupLayer(self.relation_embeddings)
        self.head_lookup = chemical_lookup(self.head)
        self.tail_lookup = disease_lookup(self.tail)
        self.rel_lookup = relation_lookup(self.rel)
        self.head_neg_lookup = chemical_lookup(self.head_neg)
        self.tail_neg_lookup = disease_lookup(self.tail_neg)

    def score_function(self, h, t, r):
        if self.score == 'l1':
            score = tf.reduce_sum(input_tensor=tf.abs(h + r - t))
        else:
            score = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(h + r - t)))
        return score

    # def _add_loss(self):
    #     score_pos = self.score_function(self.head_lookup, self.tail_lookup, self.rel_lookup)
    #     score_neg = self.score_function(self.head_neg_lookup, self.tail_neg_lookup, self.rel_lookup)
    #     self.loss = tf.reduce_sum(input_tensor=tf.maximum(0.0, 1.0 + score_pos - score_neg), name='max_margin_loss')

    def _add_model(self):
        self.model = tf.keras.Model(inputs=(self.head, self.tail, self.rel, self.head_neg, self.tail_neg),
                                    outputs=(self.head_lookup, self.tail_lookup, self.rel_lookup, self.head_neg_lookup,
                                             self.tail_neg_lookup))

    def build(self, data_train, data_val):
        self._add_inputs()
        self._load_data(data_train, data_val)
        self._add_embeddings()
        self._add_model()
        # self._add_loss()

    def next_batch(self, head, tail, rel, head_neg, tail_neg, num_batch):
        start = 0
        idx = 0
        while idx < num_batch:
            head_id = head[start:start + self.batch_size]
            tail_id = tail[start: start + self.batch_size]
            rel = rel[start: start + self.batch_size]
            head_neg = head_neg[start: start + self.batch_size]
            tail_neg = tail_neg[start: start + self.batch_size]
            start += self.batch_size
            idx += 1
            yield head_id, tail_id, rel, head_neg, tail_neg

    def _load_data(self, train_data, val_data):
        self.dataset_train = train_data
        self.dataset_val = val_data

    def train(self, early_stopping=True, patience=10):
        best_loss = 100000
        n_epoch_no_improvement = 0

        num_batch_train = len(self.dataset_train['head']) // self.batch_size + 1
        for e in range(self.epochs):
            print("\nStart of epoch %d" % (e + 1,))

            head_shuffled, tail_shuffled, rel_shuffled, head_neg_shufled, tail_neg_shuffled = shuffle(
                self.dataset_train['head'],
                self.dataset_train['tail'],
                self.dataset_train['rel'],
                self.dataset_train['head_neg'],
                self.dataset_train['tail_neg']
            )

            # Iterate over the batches of the dataset.
            for idx, batch in enumerate(self.next_batch(
                    head_shuffled, tail_shuffled, rel_shuffled, head_neg_shufled, tail_neg_shuffled, num_batch_train)):
                head, tail, rel, head_neg, tail_neg = batch
                with tf.GradientTape() as tape:
                    logits = self.model((head, tail, rel, head_neg, tail_neg), training=True)
                    # self._add_loss()
                    h, t, r, h_n, t_n = logits
                    score_pos = self.score_function(h, t, r)
                    score_neg = self.score_function(h_n, t_n, r)
                    loss_value = tf.reduce_sum(input_tensor=tf.maximum(0.0, 1.0 + score_pos - score_neg))
                    grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                if idx % 5000 == 0:
                    Log.log("Iter {}, Loss: {} ".format(idx, loss_value))

            if early_stopping:
                num_batch_val = len(self.dataset_val['head']) // self.batch_size + 1
                total_loss = []

                data = {
                    'head': self.dataset_val['head'],
                    'tail': self.dataset_val['tail'],
                    'rel': self.dataset_val['rel'],
                    'head_neg': self.dataset_val['head_neg'],
                    'tail_neg': self.dataset_val['tail_neg']
                }

                for idx, batch in enumerate(self.next_batch(data['head'], data['tail'], data['rel'], data['head_neg'],
                                                            data['tail_neg'], num_batch_val)):
                    head, tail, rel, head_neg, tail_neg = batch
                    val_logits = self.model([head, tail, rel, head_neg, tail_neg], training=True)
                    h, t, r, h_n, t_n = val_logits
                    score_pos = self.score_function(h, t, r)
                    score_neg = self.score_function(h_n, t_n, r)
                    v_loss = tf.reduce_sum(input_tensor=tf.maximum(0.0, 1.0 + score_pos - score_neg))
                    total_loss.append(float(v_loss))

                val_loss = np.mean(total_loss)
                Log.log("Loss at epoch number {}: {}".format(e + 1, val_loss))
                print("Current best loss: ", best_loss)

                if val_loss < best_loss:
                    print("New best loss: {}".format(val_loss))
                    Log.log('Save the model at epoch {}'.format(e + 1))
                    self.model.save_weights(self.model_path)
                    best_loss = val_loss
                    n_epoch_no_improvement = 0

                else:
                    n_epoch_no_improvement += 1
                    Log.log("Number of epochs with no improvement: {}".format(n_epoch_no_improvement))
                    if n_epoch_no_improvement >= patience:
                        print("Best loss: {}".format(val_loss))
                        break

        if not early_stopping:
            self.model.save_weights(self.model_path)

    def save_model(self):
        self.model.save_weights(self.model_path)

    def evaluate(self):
        distance_head_prediction = self.chemical_embeddings + self.rel_lookup - self.tail_lookup  # broadcasting
        distance_tail_prediction = self.head_lookup + self.rel_lookup - self.disease_embedings
        if self.score == 'l1':  # L1 score
            _, idx_head_prediction = tf.nn.top_k(
                tf.reduce_sum(input_tensor=tf.abs(distance_head_prediction), axis=1), k=self.chem_size)
            _, idx_tail_prediction = tf.nn.top_k(
                tf.reduce_sum(input_tensor=tf.abs(distance_tail_prediction), axis=1), k=self.rel_size)
        else:  # L2 score
            _, idx_head_prediction = tf.nn.top_k(
                tf.reduce_sum(input_tensor=tf.square(distance_head_prediction), axis=1), k=self.chem_size)
            _, idx_tail_prediction = tf.nn.top_k(
                tf.reduce_sum(input_tensor=tf.square(distance_tail_prediction), axis=1), k=self.rel_size)

        return idx_head_prediction, idx_tail_prediction
