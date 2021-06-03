import tensorflow as tf
from sklearn.utils import shuffle
import constants
import os
import numpy as np
from data_utils import count_vocab


# tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()


class TransEModel:
    def __init__(self, model_path, batch_size, epochs):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.batch_size = batch_size
        self.chem_size = count_vocab(constants.ENTITY_PATH + 'chemical2id.txt')
        self.dis_size = count_vocab(constants.ENTITY_PATH + 'disease2id.txt')
        self.rel_size = count_vocab(constants.ENTITY_PATH + 'relation2id.txt')
        self.epochs = epochs
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.dataset_train = None
        self.dataset_val = None

    def _add_inputs(self):
        self.head = tf.keras.Input(name='head', shape=(None,), dtype=tf.int32)
        self.tail = tf.keras.Input(name='tail', shape=(None,), dtype=tf.int32)
        self.rel = tf.keras.Input(name='rel', shape=(None,), dtype=tf.int32)
        self.head_neg = tf.keras.Input(name='head_neg', shape=(None,), dtype=tf.int32)
        self.tail_neg = tf.keras.Input(name='tail_neg', shape=(None,), dtype=tf.int32)

    def _add_embeddings(self):
        # generate embeddings
        self.chemical_embeddings = tf.Variable(self.initializer(shape=[self.chem_size + 1, constants.ENTITY_DIM]),
                                               dtype=tf.float32, name='chemicals', trainable=True)
        self.chemical_embeddings = tf.nn.l2_normalize(self.chemical_embeddings, axis=1)
        self.disease_embedings = tf.Variable(self.initializer(shape=[self.dis_size + 1, constants.ENTITY_DIM]),
                                             dtype=tf.float32, name='diseases', trainable=True)
        self.disease_embedings = tf.nn.l2_normalize(self.disease_embedings, axis=1)
        self.relation_embeddings = tf.Variable(self.initializer(shape=[self.rel_size + 1, constants.RELATION_DIM]),
                                               dtype=tf.float32, name='antity_relation', trainable=True)
        self.relation_embeddings = tf.nn.l2_normalize(self.relation_embeddings, axis=1)

        # lookup embedding for scoring function
        self.head_lookup = tf.nn.embedding_lookup(params=self.chemical_embeddings, ids=self.head)
        self.tail_lookup = tf.nn.embedding_lookup(params=self.disease_embedings, ids=self.tail)
        self.rel_lookup = tf.nn.embedding_lookup(params=self.relation_embeddings, ids=self.rel)
        self.head_neg_lookup = tf.nn.embedding_lookup(params=self.chemical_embeddings, ids=self.head_neg)
        self.tail_neg_lookup = tf.nn.embedding_lookup(params=self.disease_embedings, ids=self.tail_neg)

    @staticmethod
    def score_function(h, t, r):
        score = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(h + r - t), axis=1))
        return score

    def _add_loss(self):
        score_pos = self.score_function(self.head_lookup, self.tail_lookup, self.rel_lookup)
        score_neg = self.score_function(self.head_neg_lookup, self.tail_neg_lookup, self.rel_lookup)
        self.loss = tf.reduce_sum(input_tensor=tf.maximum(0.0, 1.0 + score_pos - score_neg), name='max_margin_loss',)

    def _add_model(self):
        self.model = tf.keras.Model(inputs=[self.head, self.tail, self.rel, self.head_neg, self.tail_neg],
                                    outputs=[self.head_lookup, self.tail_lookup, self.rel_lookup, self.head_neg_lookup,
                                             self.tail_neg_lookup])

    def build(self):
        self._add_inputs()
        self._add_embeddings()
        self._add_model()
        self._add_loss()

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

    def load_data(self, train_data, val_data):
        self.dataset_train = train_data
        self.dataset_val = val_data

    def train(self):
        head_shuffled, tail_shuffled, rel_shuffled, head_neg_shufled, tail_neg_shuffled = shuffle(
            self.dataset_train['head'],
            self.dataset_train['tail'],
            self.dataset_train['rel'],
            self.dataset_train['head_neg'],
            self.dataset_train['tail_neg']
        )
        num_batch_train = len(head_shuffled) // self.batch_size + 1
        for e in range(self.epochs):
            print("\nStart of epoch %d" % (e,))

            # Iterate over the batches of the dataset.
            for idx, batch in enumerate(self.next_batch(
                    head_shuffled, tail_shuffled, rel_shuffled, head_neg_shufled, tail_neg_shuffled, num_batch_train)):
                head, tail, rel, head_neg, tail_neg = batch
                with tf.GradientTape() as tape:
                    self.model([head, tail, rel, head_neg, tail_neg], training=True)
                    # self._add_loss()
                    loss_value = self.loss
                    grads = tape.gradient(loss_value, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                    # Log every 200 batches.
                    if idx % 2000 == 0:
                        print(
                            "Training loss (for one batch) at step {}:".format(idx), end=' '
                        )
                        # print(tf.numpy_function(np.asscalar, loss_value, tf.float32))
                        print(loss_value)
                        print("Seen so far: %s samples" % ((idx + 1) * self.batch_size))

    def save_model(self):
        self.model.save(self.model_path)
