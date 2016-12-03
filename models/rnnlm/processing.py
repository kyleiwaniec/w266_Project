import collections
import re
import time
import itertools
import pickle

import nltk
import nltk.data
import numpy as np
import pandas as pd

from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize

from IPython.display import display


##
# Word processing functions
def flatten(list_of_lists):
    """Flatten a list-of-lists into a single list."""
    return list(itertools.chain.from_iterable(list_of_lists))


def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word


def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "<unk>" # unknown token


def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]


def pretty_print_matrix(M, rows=None, cols=None, dtype=float):
    """Pretty-print a matrix using Pandas.

    Args:
      M : 2D numpy array
      rows : list of row labels
      cols : list of column labels
      dtype : data type (float or int)
    """
    display(pd.DataFrame(M, index=rows, columns=cols, dtype=dtype))


def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
    """Pretty-print a timedelta, using the given format string."""
    since = since or time.time()
    until = until or time.time()
    delta_s = until - since
    hours, remainder = divmod(delta_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return fmt % (hours, minutes, seconds)

##
# Data loading functions

def tokenize_sentences(s):
    tokenizer = TreebankWordTokenizer()
    return [
        tokenizer.tokenize(s)
        for s in sent_tokenize(s.decode("utf-8")[:500].strip())
    ]


def batch_generator(ids, batch_size, max_time):
    """Convert ids to data-matrix form."""
    # Clip to multiple of max_time for convenience
    clip_len = ((len(ids)-1) / batch_size) * batch_size
    input_w = ids[:clip_len]     # current word
    target_y = ids[1:clip_len+1]  # next word
    # Reshape so we can select columns
    input_w = input_w.reshape([batch_size,-1])
    target_y = target_y.reshape([batch_size,-1])

    # Yield batches
    for i in xrange(0, input_w.shape[1], max_time):
    	yield input_w[:,i:i+max_time], target_y[:,i:i+max_time]


class Vocabulary(object):
    START_TOKEN = "<s>"
    END_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"

    def __init__(self, tokens, size=None):
        self.unigram_counts = collections.Counter(tokens)
        # leave space for "<s>", "</s>", and "<unk>"
        top_counts = self.unigram_counts.most_common(None if size is None else (size - 3))
        vocab = ([self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN] +
                 [w for w,c in top_counts])

        # Assign an id to each word, by frequency
        self.id_to_word = dict(enumerate(vocab))
        self.word_to_id = {v:k for k,v in self.id_to_word.iteritems()}
        self.size = len(self.id_to_word)
        if size is not None:
            assert(self.size <= size)

        # Store special IDs
        self.START_ID = self.word_to_id[self.START_TOKEN]
        self.END_ID = self.word_to_id[self.END_TOKEN]
        self.UNK_ID = self.word_to_id[self.UNK_TOKEN]

    def words_to_ids(self, words):
        return [self.word_to_id.get(w, self.UNK_ID) for w in words]

    def ids_to_words(self, ids):
        return [self.id_to_word[i] for i in ids]

    def sentence_to_ids(self, words):
        return [self.START_ID] + self.words_to_ids(words) + [self.END_ID]

    def ordered_words(self):
        """Return a list of words, ordered by id."""
        return self.ids_to_words(range(self.size))


class Corpus(object):
    def __init__(self, content, V):
        self.V = V

        print "Building vocabulary..."

        tokenizer = TreebankWordTokenizer()
        token_feed = [
            canonicalize_word(w)
            for s in content
            for w in tokenizer.tokenize(s)
        ]

        self.vocab = Vocabulary(token_feed, size=V)
        print "Done."

    def preprocess_sentences(self, sentences):
        # Add sentence boundaries, canonicalize, and handle unknowns
        words = ["<s>"] + flatten(s + ["<s>"] for s in sentences)
        words = [canonicalize_word(w, wordset=self.vocab.word_to_id)
                 for w in words]
        return np.array(self.vocab.words_to_ids(words))

    def generate_training_data(self, content, train_frac=0.8):
        print "Finding sentences..."

        tokenizer = TreebankWordTokenizer()
        sentences = np.array([
            [canonicalize_word(w) for w in sent]
            for s in content
            for sent in tokenize_sentences(s)
        ], dtype=object)

        print "Processing sentences..."

        fmt = (len(sentences), sum(map(len, sentences)))
        print "Loaded %d sentences (%g tokens)" % fmt

        rng = np.random.RandomState(42)
        rng.shuffle(sentences)  # in-place

        split_idx = int(train_frac * len(sentences))
        train_sentences = sentences[:split_idx]
        test_sentences = sentences[split_idx:]

        fmt = (len(train_sentences), sum(map(len, train_sentences)))
        print "Training set: %d sentences (%d tokens)" % fmt

        fmt = (len(test_sentences), sum(map(len, test_sentences)))
        print "Test set: %d sentences (%d tokens)" % fmt

        training_data = {
            "train_ids": self.preprocess_sentences(train_sentences),
            "test_ids": self.preprocess_sentences(test_sentences),
        }

        print "Done."
        return training_data
