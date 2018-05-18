# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import os
import random

import sys

from tensorflow.python.framework.graph_util import convert_variables_to_constants
from seq2word_rnn_model import WordModel, LetterModel
from config import Config
import config
import data_feeder as data_feeder


FLAGS = config.FLAGS
data_type = config.data_type
index_data_type = config.index_data_type
np_index_data_type = config.np_index_data_type


def export_graph(session, iter, phase="lm"):
    if phase == "lm":
        # Export variables related to language model only
        variables_to_export = ["Online/WordModel/probabilities",
                               "Online/WordModel/state_out",
                               # "Online/WordModel/phrase_p_probabilities",
                               # "Online/WordModel/phrase_p_prediction",
                               # "Online/WordModel/phrase_probabilities",
                               # "Online/WordModel/phrase_top_k_prediction",
                               # "Online/WordModel/logits_phrase",
                               "Online/WordModel/top_k_prediction"]
    elif phase == "kc_full":
        variables_to_export = ["Online/WordModel/probabilities",
                               "Online/WordModel/state_out",
                               "Online/WordModel/phrase_p_probabilities",
                               "Online/WordModel/phrase_p_prediction",
                               "Online/WordModel/phrase_probabilities",
                               "Online/WordModel/phrase_top_k_prediction",
                               "Online/WordModel/logits_phrase",
                               "Online/WordModel/top_k_prediction",
                               "Online/LetterModel/probabilities",
                               "Online/LetterModel/state_out",
                               "Online/LetterModel/top_k_prediction"]
    else:
        assert phase == "kc_slim"
        variables_to_export = ["Online/WordModel/state_out",
                               # "Online/WordModel/phrase_p_probabilities",
                               # "Online/WordModel/phrase_p_prediction",
                               # "Online/WordModel/logits_phrase",
                               "Online/LetterModel/probabilities",
                               "Online/LetterModel/state_out",
                               "Online/LetterModel/top_k_prediction"]

    graph_def = convert_variables_to_constants(session, session.graph_def, variables_to_export)
    config_name = FLAGS.model_config
    model_export_path = os.path.join(FLAGS.graph_save_path)
    if not os.path.isdir(model_export_path):
        os.makedirs(model_export_path)
    model_export_name = os.path.join(model_export_path,
                                     config_name[config_name.rfind("/")+1:] + "-iter" + str(iter) + "-" + phase + '.pb')
    f = open(model_export_name, "wb")
    f.write(graph_def.SerializeToString())
    f.close()
    print("Graph is saved to: ", model_export_name)


def run_letter_epoch(session, data, word_model, letter_model, config, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    num_word = 0
    fetches = {}
    fetches_letter = {}
    previous_state = session.run(word_model.initial_state)

    for step, (epoch_size, lm_data, letter_data, phrase_p_data, phrase_data) in \
            enumerate(data_feeder.data_iterator(data, config)):
        if FLAGS.laptop_discount > 0 and step >= FLAGS.laptop_discount:
            break
        if step >= epoch_size:
            break

        fetches["rnn_state"] = word_model.rnn_state
        fetches["final_state"] = word_model.final_state
        fetches_letter["cost"] = letter_model.cost

        if eval_op is not None:
            fetches_letter["eval_op"] = eval_op
        feed_dict = {word_model.input_data: lm_data[0],
                     word_model.target_data: lm_data[1],
                     word_model.output_masks: lm_data[2],
                     word_model.sequence_length: lm_data[3],
                     word_model.initial_state: previous_state}
        vals = session.run(fetches, feed_dict)

        previous_state = vals["final_state"]
        rnn_state_to_letter_model = vals["rnn_state"]

        feed_dict_letter = {letter_model.lm_state_in: rnn_state_to_letter_model,
                            letter_model.input_data: letter_data[0],
                            letter_model.target_data: letter_data[1],
                            letter_model.output_masks: letter_data[2],
                            letter_model.sequence_length: letter_data[3]}
        vals_letter = session.run(fetches_letter, feed_dict_letter)
        cost_letter = vals_letter["cost"]

        costs += cost_letter
        iters += np.sum(letter_data[2])
        num_word += np.sum(letter_data[3])

        if verbose and step % (epoch_size // 100) == 0:
            if costs / iters > 100.0:#因为在算交叉熵的时候，是按照mask给对应的label加了权重的，所以最后求平均的时候也要除以这些权重的和，即mask的和。
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] PPL TOO LARGE! %.3f ENTROPY: (%.3f) speed: %.0f wps" %
                      (step * 1.0 / epoch_size, costs / iters, num_word / (time.time() - start_time)))
            else:
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] %.3f letter_ppl: %.3f speed: %.0f wps" %
                      (step * 1.0 / epoch_size, np.exp(costs / iters), num_word / (time.time() - start_time)))
            sys.stdout.flush()

    return np.exp(costs / iters)#PPL


def run_word_epoch(session, data, word_model, config, lm_phase_id, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    costs_phrase_p =0
    costs_phrase = 0
    iters_phrase_p = 0
    iters_phrase = 0
    num_word = 0
    fetches = {}
    previous_state = session.run(word_model.initial_state)

    #for step, (epoch_size, lm_data, _, phrase_p_data, phrase_data) in \
    for step, (epoch_size, lm_data) in \
                    enumerate(data_feeder.data_iterator(data, config)):
        if FLAGS.laptop_discount > 0 and step >= FLAGS.laptop_discount:
            break
        if step >= epoch_size:
            break

        fetches["cost"] = word_model.cost
        fetches["final_state"] = word_model.final_state

        if eval_op is not None:
            fetches["eval_op"] = eval_op[lm_phase_id]

        feed_dict = {word_model.input_data: lm_data[0],
                     word_model.target_data: lm_data[1],
                     word_model.output_masks: lm_data[2],
                     word_model.sequence_length: lm_data[3],
                     # word_model.target_phrase_p: phrase_p_data[0],
                     # word_model.target_phrase_p_masks: phrase_p_data[1],
                     # word_model.target_phrase_data: phrase_data[0],
                     # word_model.target_phrase_data_masks: phrase_data[1],
                     # word_model.target_phrase_logits_masks: phrase_data[2],
                     word_model.initial_state: previous_state
                     }
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        previous_state = vals["final_state"]

        costs += cost[0]
        # costs_phrase_p += cost[1]
        # costs_phrase += cost[2]
        iters += np.sum(lm_data[2])
        # iters_phrase_p += np.sum(phrase_p_data[1])
        # iters_phrase += np.sum(phrase_data[1])

        num_word += np.sum(lm_data[3])
        if verbose and step % (epoch_size // 100) == 0:
            # print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] "
            #       "%.3f word ppl: %.3f phrase prob ppl: %.3f phrase ppl: %.3f speed: %.0f wps"
            #       % (step * 1.0 / epoch_size, np.exp(costs / iters),
            #       np.exp(costs_phrase_p / iters_phrase_p),
            #       np.exp(costs_phrase / iters_phrase),
            #       num_word / (time.time() - start_time)))
            print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] "
                                                             "%.3f word ppl: %.3f speed: %.0f wps"
                  % (step * 1.0 / epoch_size, np.exp(costs / iters),
                     num_word / (time.time() - start_time)))
            sys.stdout.flush()
    #all_costs = np.exp(costs / iters), np.exp(costs_phrase_p / iters_phrase_p), np.exp(costs_phrase / iters_phrase)
    all_costs = np.exp(costs / iters)
    #return all_costs[lm_phase_id]
    return all_costs


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    
    logfile = open(FLAGS.model_config + '.log', 'w')

    config = Config()
    config.get_config(FLAGS.vocab_path, FLAGS.model_config)

    test_config = Config()
    test_config.get_config(FLAGS.vocab_path, FLAGS.model_config)
    test_config.batch_size = 1
    test_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_fraction
        train_data = data_feeder.read_file(FLAGS.data_path, config, is_train=True)
        valid_data = data_feeder.read_file(FLAGS.data_path, config, is_train=False)
        print("in words vocabulary size = %d\nout words vocabulary size = %d\nin letters vocabulary size = %d"
              "\nphrase vocabulary size = %d" % (
                  config.vocab_size_in, config.vocab_size_out, config.vocab_size_letter,
                  config.vocab_size_phrase))

        with tf.Session(config=gpu_config) as session:
            with tf.name_scope("Train"):

                with tf.variable_scope("WordModel", reuse=False, initializer=initializer):
                    mtrain = WordModel(is_training=True, config=config)
                    train_op = mtrain.train_op

                with tf.variable_scope("LetterModel", reuse=False, initializer=initializer):
                    mtrain_letter = LetterModel(is_training=True, config=config)
                    train_letter_op = mtrain_letter.train_op

            with tf.name_scope("Valid"):

                with tf.variable_scope("WordModel", reuse=True, initializer=initializer):
                    mvalid = WordModel(is_training=False, config=config)

                with tf.variable_scope("LetterModel", reuse=True, initializer=initializer):
                    mvalid_letter = LetterModel(is_training=False, config=config)
            with tf.name_scope("Online"):
                
                with tf.variable_scope("WordModel", reuse=True, initializer=initializer):
                    monline = WordModel(is_training=False, config=test_config)
                with tf.variable_scope("LetterModel", reuse=True, initializer=initializer):
                    monline_letter = LetterModel(is_training=False, config=test_config)

            restore_variables = dict()
            for v in tf.trainable_variables():

                print("store:", v.name)
                # if v.name.startswith("atten_"):
                #     continue
                restore_variables[v.name] = v
            sv = tf.train.Saver(restore_variables)#创建一个Saver实例，指定了将要保存和恢复的变量。它可以传dict 或者list

            if not FLAGS.model_name.endswith(".ckpt"):
                FLAGS.model_name += ".ckpt"

            session.run(tf.global_variables_initializer())
            
            check_point_dir = os.path.join(FLAGS.save_path)
            ckpt = tf.train.get_checkpoint_state(check_point_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                sv.restore(session, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")

            print("training language model.")
            print("training language model", file=logfile)#直接打印到文件中。需要先open这个文件

            save_path = os.path.join(FLAGS.save_path)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            for lm_phase_id in range(1):#(cost, cost_phrase_p, cost_phrase)
                # if lm_phase_id != 1:
                #     continue
                # ln_phase_id = [0,1,2], representing lm_train_phase, phrase_prob_train_phase and phrase_train_phase
                print("lm training phase: %d" % (lm_phase_id + 1), file=logfile)
                if lm_phase_id != 1:
                    max_train_epoch = config.max_max_epoch
                else:#lm_phrase_id == 1是对phrase_p训练
                    max_train_epoch = 2
                # the max_train_epoch of phrase_prob_train_phase is 2, it is enough.
                for i in range(max_train_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0)
                    print("lm training phase: %d" % (lm_phase_id + 1))
                    mtrain.assign_lr(session, config.learning_rate * lr_decay)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)), file=logfile)
                    train_perplexity = run_word_epoch(session, train_data, mtrain, config, lm_phase_id, train_op, verbose=True)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity), file=logfile)
                    logfile.flush()#刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入。

                    valid_perplexity = run_word_epoch(session, valid_data, mvalid, config, lm_phase_id)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity), file=logfile)
                    logfile.flush()

                    if FLAGS.save_path:
                        print("Saving model to %s." % FLAGS.save_path, file=logfile)
                        step = mtrain.get_global_step(session)
                        model_save_path = os.path.join(save_path, FLAGS.model_name)
                        sv.save(session, model_save_path, global_step=step)
                    print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin exporting lm graph!")
                    export_graph(session, i, phase="lm")
                    export_graph(session, i, phase="kc_slim")
                    print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Finish exporting lm graph!")
                
            # print("training letter model.")
            # print("training letter model", file=logfile)
            # for i in range(config.max_max_epoch):
            #     lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0)
            #
            #     mtrain_letter.assign_lr(session, config.learning_rate * lr_decay)
            #     print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
            #     print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain_letter.lr)), file=logfile)
            #     train_perplexity = run_letter_epoch(session,train_data, mtrain, mtrain_letter, config, train_letter_op,
            #                                         verbose=True)
            #     print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
            #     print("Epoch: %d Train ppl: %.3f" % (i + 1, train_perplexity), file=logfile)
            #     logfile.flush()
            #
            #     print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
            #     valid_perplexity = run_letter_epoch(session, valid_data, mtrain, mvalid_letter, config)
            #     print("Epoch: %d Valid ppl: %.3f" % (i + 1, valid_perplexity), file=logfile)
            #     logfile.flush()
            #
            #     print("Saving model to %s." % FLAGS.save_path, file=logfile)
            #     step = mtrain_letter.get_global_step(session)
            #     model_save_path = os.path.join(save_path, FLAGS.model_name)
            #     sv.save(session, model_save_path, global_step=step)
            #     print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin exporting letter model graph!")
            #
            #     export_graph(session, i, phase="kc_full")
            #     export_graph(session, i, phase="kc_slim")
            #
            #     print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Finish exporting letter model graph!")
            
            logfile.close()

if __name__ == "__main__":
    tf.app.run()
