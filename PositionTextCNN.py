import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
import pandas as pd


class PositionTextCNN(object):
    """
    A Basic CNN for text classification with Position features as well.
    
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    
    Refer tohttps://www.ncbi.nlm.nih.gov/pmc/articles/PMC5181565/pdf/btw486.pdf for more details.
    """
    def __init__(self, sequence_length, vocab_processor, 
                 num_classes=2, embedding_size=128, filter_sizes=[3,4,5], 
                 num_filters=128, batch_size=64, 
                 l2_reg_lambda=0.0, num_epochs=200,
                 num_checkpoints=5, dropout_prob=0.5, 
                 checkpoint_every=100, evaluate_every=100, 
                 allow_soft_placement=True,log_device_placement=False,
                 results_dir="runs"):
        
        tf.reset_default_graph() 
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = len(vocab_processor.vocabulary_)
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.num_epochs = num_epochs
        self.results_dir = results_dir
        
        self.vocab_processor = vocab_processor
        
        self.num_checkpoints = num_checkpoints
        self.dropout_prob = dropout_prob
        self.checkpoint_every = checkpoint_every
        self.evaluate_every = evaluate_every
        
        self.position_vector_mapping = PositionTextCNN.load_position_vector_mapping()
        
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        
        self.sess = tf.Session()
        self._build_network()
        
    def _build_network(self):
        
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        self.word_distancesA = tf.placeholder(tf.int32, [None, self.sequence_length], name="train_word_distancesA")
        self.word_distancesB = tf.placeholder(tf.int32, [None, self.sequence_length], name="train_word_distancesB")

        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("word_embedding"):
            self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W") 
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            
        # Position Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("position_embedding"):
            embedded_positionsA = tf.nn.embedding_lookup(self.position_vector_mapping, self.word_distancesA)
            embedded_positionsB = tf.nn.embedding_lookup(self.position_vector_mapping, self.word_distancesB)
            embedded_positions = tf.concat([embedded_positionsA, embedded_positionsB], 2)
            self.embedded_positions_expanded = tf.cast(tf.expand_dims(embedded_positions, -1), tf.float32)
            
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.final_embedded_expanded = tf.concat([self.embedded_chars_expanded, self.embedded_positions_expanded], 2)
        
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size+20, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.final_embedded_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            
    def train_network(self, x_train, y_train, x_dev, y_dev,
                     train_word_distancesA, train_word_distancesB, test_word_distancesA, test_word_distancesB):
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            class_weight = tf.constant([1.0, 100.0])
            weights = tf.reduce_sum(class_weight * self.input_y, axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            weighted_losses = losses * weights
            self.loss = tf.reduce_mean(weighted_losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            _, self.precision = tf.metrics.precision(labels=tf.argmax(self.input_y, 1), predictions=self.predictions, name='precision')
            _, self.recall = tf.metrics.recall(labels=tf.argmax(self.input_y, 1), predictions=self.predictions, name='recall')
            
        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, self.results_dir, timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        precision_summary = tf.summary.scalar("precision", self.precision)
        recall_summary = tf.summary.scalar("recall", self.recall)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, precision_summary, recall_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

        # Dev summaries
        self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary, precision_summary, recall_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)
        
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)
        
        # Write vocabulary
        self.vocab_processor.save(os.path.join(out_dir, "vocab"))
        
        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
    
        print("Start training")
        # Generate batches
        batches = PositionTextCNN.batch_iter(
            list(zip(x_train, y_train, train_word_distancesA, train_word_distancesB)), self.batch_size, self.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch, batch_word_distancesA, batch_word_distancesB = zip(*batch)
            self.train_step(x_batch, y_batch, batch_word_distancesA, batch_word_distancesB)
            current_step = tf.train.global_step(self.sess, self.global_step)
            if current_step % self.evaluate_every == 0:
                print("\nEvaluation:")
                self.dev_step(x_dev, y_dev, test_word_distancesA, test_word_distancesB, writer=self.dev_summary_writer)
                print("")
            if current_step % self.checkpoint_every == 0:
                path = saver.save(self.sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))   
        print("Training finished")
    
    def train_step(self, x_batch, y_batch, batch_word_distancesA, batch_word_distancesB):
        """
        A single training step
        """
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: self.dropout_prob,
            self.word_distancesA: batch_word_distancesA,
            self.word_distancesB: batch_word_distancesB,
        }
        _, step, summaries, loss, accuracy, precision, recall = self.sess.run(
            [self.train_op, self.global_step, self.train_summary_op, self.loss, self.accuracy, self.precision, self.recall],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}, prec {:g}, recall {:g}".format(time_str, step, loss, accuracy, precision, recall))
        self.train_summary_writer.add_summary(summaries, step)
        
    
    def dev_step(self, x_batch, y_batch, batch_word_distancesA, batch_word_distancesB, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              self.input_x: x_batch,
              self.input_y: y_batch,
              self.dropout_keep_prob: 1.0,
              self.word_distancesA: batch_word_distancesA,
              self.word_distancesB: batch_word_distancesB,
            }
            step, summaries, loss, accuracy,  precision, recall  = self.sess.run(
                [self.global_step, self.dev_summary_op, self.loss, self.accuracy, self.precision, self.recall],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, prec {:g}, recall {:g}".format(time_str, step, loss, accuracy, precision, recall))
            if writer:
                writer.add_summary(summaries, step)
                
    @staticmethod            
    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
    
    @staticmethod 
    def load_position_vector_mapping():
        # bit_array generated with the distance between 
        # two entities where abs_num represents the distance
        def int2bit_by_distance(int_num, bit_len=10):

            bit_array = np.zeros(bit_len)
            if int_num > 0:
                bit_array[0] = 1

            abs_num = np.abs(int_num)
            if abs_num <= 5:
                for i in range(abs_num):
                    bit_array[-i-1] = 1
            elif abs_num <= 10:
                for i in range(6):
                    bit_array[-i-1] = 1
            elif abs_num <= 20:
                for i in range(7):
                    bit_array[-i-1] = 1
            elif abs_num <= 30:
                for i in range(8):
                    bit_array[-i-1] = 1
            else:
                for i in range(9):
                    bit_array[-i-1] = 1
            return bit_array

        map = {}
        for i in range(-300, 300):
            map[i] = int2bit_by_distance(i, 10)

        return pd.DataFrame.from_dict(map, orient='index', dtype='int').values