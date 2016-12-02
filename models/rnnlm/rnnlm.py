import time

import tensorflow as tf
import numpy as np


import processing


def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.

    Args:
    X: [m,n,k]
    W: [k,l]

    Returns:
    XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


def MakeFancyRNNCell(H, keep_prob, num_layers=1):
    """Make a fancy RNN cell.

    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.

    Args:
    H: hidden state size
    keep_prob: dropout keep prob (same for input and output)
    num_layers: number of cell layers

    Returns:
    (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob,
                                       output_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
    return cell


def load_and_score(inputs, corpus, model_params, trained_filename, sort=False):
    """Load the trained model and score the given words."""
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            lm = RNNLM(model_params)
            lm.BuildCoreGraph()

        # Load the trained model
        saver = tf.train.Saver()
        saver.restore(session, trained_filename)

        if isinstance(inputs[0], str) or isinstance(inputs[0], unicode):
            inputs = [inputs]

        # Actually run scoring
        results = []
        for words in inputs:
            score = lm.ScoreSeq(session, words, corpus.vocab)
            results.append((score, words))

        # Sort if requested
        if sort: results = sorted(results, reverse=True)

        # Print results
        for score, words in results:
            print "\"%s\" : %.02f" % (" ".join(words), score)

            
def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
    """Pretty-print a timedelta, using the given format string."""
    since = since or time.time()
    until = until or time.time()
    delta_s = until - since
    hours, remainder = divmod(delta_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return fmt % (hours, minutes, seconds)


class RNNParams(object):
    def __init__(self, V, H, num_layers):
        self.V = V
        self.H = H
        self.num_layers = num_layers

class RNNLM(object):

    def __init__(self, params):
        """Init function.

        This function just stores hyperparameters. You'll do all the real graph
        construction in the Build*Graph() functions below.

        Args:
          V: vocabulary size
          H: hidden state dimension
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Model structure; these need to be fixed for a given model.
        self.V = params.V
        self.H = params.H
        self.num_layers = params.num_layers

        # Training hyperparameters; these can be changed with feed_dict,
        # and you may want to do so during training.
        with tf.name_scope("Training_Parameters"):
            self.learning_rate_ = tf.constant(0.1, name="learning_rate")
            self.dropout_keep_prob_ = tf.placeholder(tf.float32, name="dropout_keep_prob")
            # For gradient clipping, if you use it.
            # Due to a bug in TensorFlow, this needs to be an ordinary python
            # constant instead of a tf.constant.
            self.max_grad_norm_ = 5.0

    def BuildCoreGraph(self):
        """Construct the core RNNLM graph, needed for any use of the model.

        This should include:
        - Placeholders for input tensors (input_w_, initial_h_, target_y_)
        - Variables for model parameters
        - Tensors representing various intermediate states
        - A Tensor for the final state (final_h_)
        - A Tensor for the output logits (logits_), i.e. the un-normalized argument
          of the softmax(...) function in the output layer.
        - A scalar loss function (loss_)

        Your loss function should return a *scalar* value that represents the
        _summed_ loss across all examples in the batch (i.e. use tf.reduce_sum, not
        tf.reduce_mean).

        You shouldn't include training or sampling functions here; you'll do this
        in BuildTrainGraph and BuildSampleGraph below.

        We give you some starter definitions for input_w_ and target_y_, as well
        as a few other tensors that might help. We've also added dummy values for
        initial_h_, logits_, and loss_ - you should re-define these in your code as
        the appropriate tensors. See the in-line comments for more detail.
        """
        # Input ids, with dynamic shape depending on input.
        # Should be shape [batch_size, max_time] and contain integer word indices.
        self.input_w_ = tf.placeholder(tf.int32, [None, None], name="w")

        # Should be the same shape as inputs_w_
        self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")

        # Get dynamic shape info from inputs
        with tf.name_scope("batch_size"):
            self.batch_size_ = tf.shape(self.input_w_)[0]
        with tf.name_scope("max_time"):
            self.max_time_ = tf.shape(self.input_w_)[1]

        # Get sequence length from input_w_.
        # This will be a vector with elements ns[i] = len(input_w_[i])
        # You can override this in feed_dict if you want to have different-length
        # sequences in the same batch, although you shouldn't need to for this
        # assignment.
        self.ns_ = tf.tile([self.max_time_], [self.batch_size_,], name="ns")

        # Construct embedding layer
        embed_weights = tf.Variable(
            tf.random_uniform([self.V, self.H], -1.0, 1.0),
            name="embeddings")

        embedding = tf.nn.embedding_lookup(embed_weights, self.input_w_)

        # Construct RNN/LSTM cell and recurrent layer (hint: use tf.nn.dynamic_rnn)
        cell = MakeFancyRNNCell(self.H, self.dropout_keep_prob_, self.num_layers)
        self.initial_h_ = cell.zero_state(self.batch_size_, tf.float32)
        self.rnn_outputs_, self.final_h_ = tf.nn.dynamic_rnn(
            cell, embedding, initial_state=self.initial_h_)

        # Softmax output layer, over vocabulary
        # Hint: use the matmul3d() helper here.
        self.output_weights_ = tf.Variable(
            tf.random_uniform([self.V, self.H], -1.0, 1.0),
            name="output_weights")
        self.output_biases_ = tf.Variable(
            tf.random_uniform([self.V], -1.0, 1.0),
            name="output_biases")

        self.output_ = (
            matmul3d(self.rnn_outputs_, tf.transpose(self.output_weights_)) +
            self.output_biases_)

        # Loss computation (true loss, for prediction)
        self.loss_ = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.output_, self.target_y_))

    def BuildTrainGraph(self):
        """Construct the training ops.

        You should define:
        - train_loss_ (optional): an approximate loss function for training
        - train_step_ : a training op that can be called once per batch

        Your loss function should return a *scalar* value that represents the
        _summed_ loss across all examples in the batch (i.e. use tf.reduce_sum, not
        tf.reduce_mean).
        """
        # Replace this with an actual training op
        self.train_step_ = tf.no_op(name="dummy")

        # Replace this with an actual loss function
        self.train_loss_ = None

        # Define loss function(s)
        with tf.name_scope("Train_Loss"):
            self.train_loss_ = tf.reduce_sum(
                tf.nn.sampled_softmax_loss(
                    self.output_weights_, self.output_biases_,
                    tf.reshape(self.rnn_outputs_, [-1, self.H]),
                    tf.reshape(self.target_y_, [-1, 1]),
                    1000, self.V)
              )

        # Define optimizer and training op
        with tf.name_scope("Training"):
            self.train_step_ = (
                tf.train.AdagradOptimizer(self.learning_rate_)
                .minimize(self.train_loss_))

    def BuildSamplerGraph(self):
        """Construct the sampling ops.

        You should define pred_samples_ to be a Tensor of integer indices for
        sampled predictions for each batch element, at each timestep.

        Hint: use tf.multinomial, along with a couple of calls to tf.reshape
        """
        # Replace with a Tensor of shape [batch_size, max_time, 1]
        self.pred_samples_ = tf.reshape(
            tf.multinomial(
                tf.reshape(self.output_, [self.batch_size_ * self.max_time_, -1]),
                1),
            [self.batch_size_, self.max_time_, 1])

    def RunEpoch(self, session, batch_iterator, train=False,
                  verbose=False, tick_s=10,
                  keep_prob=1.0, learning_rate=0.1):
        start_time = time.time()
        tick_time = start_time  # for showing status
        total_cost = 0.0  # total cost, summed over all words
        total_words = 0

        if train:
            train_op = self.train_step_
            keep_prob = keep_prob
            loss = self.train_loss_
        else:
            train_op = tf.no_op()
            keep_prob = 1.0  # no dropout at test time
            loss = self.loss_  # true loss, if train_loss is an approximation

        for i, (w, y) in enumerate(batch_iterator):
            cost = 0.0

            # At first batch in epoch, get a clean initial state
            if i == 0:
                h = session.run(self.initial_h_, {self.input_w_: w})

            _, cost, h = session.run([self.train_step_, self.train_loss_, self.final_h_], feed_dict={
                self.input_w_: w,
                self.target_y_: y,
                self.initial_h_: h,
                self.learning_rate_: learning_rate,
                self.dropout_keep_prob_: keep_prob,
            })

            total_cost += cost
            total_words += w.size  # w.size = batch_size * max_time

            ##
            # Print average loss-so-far for epoch
            # If using train_loss_, this may be an underestimate.
            if verbose and (time.time() - tick_time >= tick_s):
                avg_cost = total_cost / total_words
                avg_wps = total_words / (time.time() - start_time)
                print "[batch %d]: seen %d words at %d wps, loss = %.3f" % (i,
                total_words, avg_wps, avg_cost)
                tick_time = time.time()  # reset time ticker

        return total_cost / total_words

    def ScoreDataset(self, session, ids, name="Data"):
        bi = processing.batch_generator(ids, batch_size=100, max_time=100)
        cost = self.RunEpoch(session, bi,
                             learning_rate=1.0, keep_prob=1.0,
                             train=False, verbose=False, tick_s=3600)
        print "%s: avg. loss: %.03f  (perplexity: %.02f)" % (name, cost, np.exp(cost))

    def ScoreSeq(self, session, seq, vocab):
        """Score a sequence of words. Returns total log-probability."""
        padded_ids = vocab.words_to_ids(processing.canonicalize_words(
            ["<s>"] + seq, wordset=vocab.word_to_id))

        w = np.reshape(padded_ids[:-1], [1,-1])
        y = np.reshape(padded_ids[1:],  [1,-1])
        h = session.run(self.initial_h_, {self.input_w_: w})
        feed_dict = {self.input_w_:w,
                     self.target_y_:y,
                     self.initial_h_:h,
                     self.dropout_keep_prob_: 1.0}
        # Return log(P(seq)) = -1*loss
        return -1*session.run(self.loss_, feed_dict)
