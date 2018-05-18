import tensorflow as tf
import numpy as np
import os
from data_utility import DataUtility

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model_config", "static-sanity-check.cfg", "Model hyperparameters configuration. See smallconfig.cfg for an example.")
flags.DEFINE_string("data_path", "resource/train_data/", "Where the training/test data is stored.")
flags.DEFINE_string("vocab_path", "resource/vocab/", "Where the training/test data is stored.")
flags.DEFINE_string("file_name_stem", "ptb", "corpus file name (without suffix).")
flags.DEFINE_string("num_test_cases", 0, "# of test cases reading from stdin. Works for inference phase only. Default: zero.")
flags.DEFINE_string("save_path", "model/", "Model output directory.")
flags.DEFINE_string("model_name", "model_test", "Pick a name for your model.")
flags.DEFINE_string("graph_save_path", "graph/", "Exported graph directory.")
flags.DEFINE_integer("laptop_discount", -1, "Run less iterations on laptop.")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("cpu_count", 2, "# of cpus to calculate sparse representation in parallel.")
FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


def index_data_type():
    return tf.int32


def np_index_data_type():
    return np.int32


class Config():
    def __init__(self):
        self.init_scale = 0.05
        self.learning_rate = 1.0
        self.max_grad_norm = 5
        self.num_layers = 2
        self.num_steps = 35
        self.max_word_length = 30
        self.word_embedding_size = 40
        self.letter_embedding_size = 40
        self.word_hidden_size = 200
        self.letter_hidden_size = 200
        self.max_epoch = 4
        self.keep_prob = 1.0
        self.lr_decay = 0.8
        self.vocab_size_letter = 27
        self.vocab_size_in = 5000
        self.vocab_size_out = 5000
        self.vocab_size_phrase = 2000
        self.max_max_epoch = 16
        self.gpu_fraction = 0.32

    def get_config(self, vocab_path, config_filename=None):
        vocab_file_in_words = os.path.join(vocab_path, "vocab_in_words")
        vocab_file_in_letters = os.path.join(vocab_path, "vocab_in_letters")
        vocab_file_out = os.path.join(vocab_path, "vocab_out")
        vocab_file_phrase = os.path.join(vocab_path, "vocab_phrase")

        self.data_utility = DataUtility(vocab_file_in_words=vocab_file_in_words,
                                        vocab_file_in_letters=vocab_file_in_letters,
                                        vocab_file_out=vocab_file_out,
                                        vocab_file_phrase=vocab_file_phrase)

        self.vocab_size_letter = self.data_utility.in_letters_count
        self.vocab_size_in = self.data_utility.in_words_count
        self.vocab_size_out = self.data_utility.out_words_count
        self.vocab_size_phrase = self.data_utility.phrase_count

        if config_filename is not None:
            with open(config_filename) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    param, value = line.split()
                    if param == "init_scale":
                        self.init_scale = float(value)
                    elif param == "learning_rate":
                        self.learning_rate = float(value)
                    elif param == "max_grad_norm":
                        self.max_grad_norm = float(value)
                    elif param == "num_layers":
                        self.num_layers = int(value)
                    elif param == "num_steps":
                        self.num_steps = int(value)
                    elif param == "max_word_length":
                        self.max_word_length = int(value)
                    elif param == "word_embedding_size":
                        self.word_embedding_size = int(value)
                    elif param == "letter_embedding_size":
                        self.letter_embedding_size = int(value)
                    elif param == "word_hidden_size":
                        self.word_hidden_size = int(value)
                    elif param == "letter_hidden_size":
                        self.letter_hidden_size = int(value)
                    elif param == "max_epoch":
                        self.max_epoch = int(value)
                    elif param == "max_max_epoch":
                        self.max_max_epoch = int(value)
                    elif param == "keep_prob":
                        self.keep_prob = float(value)
                    elif param == "lr_decay":
                        self.lr_decay = float(value)
                    elif param == "batch_size":
                        self.batch_size = int(value)
                    elif param == "gpu_fraction":
                        self.gpu_fraction = float(value)

