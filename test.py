#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import time
from data_utility import DataUtility
from config import Config
import tensorflow as tf
import numpy as np


class InputEngineRnn:

    def __init__(self, graph_file, vocab_path, full_vocab, config_name):

        vocab_file_in_words = os.path.join(vocab_path, "vocab_in_words")
        vocab_file_in_letters = os.path.join(vocab_path, "vocab_in_letters")
        vocab_file_out = os.path.join(vocab_path, "vocab_out")
        vocab_file_phrase = os.path.join(vocab_path, "vocab_phrase")

        self._config = Config()
        self._config.get_config(vocab_path, config_name)
        self._data_utility = DataUtility(vocab_file_in_words=vocab_file_in_words, vocab_file_in_letters=vocab_file_in_letters,
                                         vocab_file_out=vocab_file_out, vocab_file_phrase=vocab_file_phrase,
                                         full_vocab_file_in_words=full_vocab)
        print("in words vocabulary size = %d\nout words vocabulary size = %d\nin letters vocabulary size = %d"
              "\nphrase vocabulary size = %d" % (
                self._config.vocab_size_in, self._config.vocab_size_out, self._config.vocab_size_letter,
                self._config.vocab_size_phrase))
        
        prefix = "import/"
        self.lm_state_in_name = prefix + "Online/WordModel/state:0"
        self.lm_input_name = prefix + "Online/WordModel/batched_input_word_ids:0"
        self.lm_state_out_name = prefix + "Online/WordModel/state_out:0"
        # self.lm_top_k_name = prefix + "Online/WordModel/top_k:0"

        self.phrase_p_name = prefix + "Online/WordModel/phrase_p_prediction: 1"
        self.phrase_p_probability = prefix + "Online/WordModel/phrase_p_probabilities: 0"
        self.phrase_top_k_name = prefix + "Online/WordModel/phrase_top_k_prediction: 1"
        self.phrase_top_k_probability = prefix + "Online/WordModel/phrase_probabilities: 0"
        self.phrase_logits = prefix + "Online/WordModel/logits_phrase: 0"

        self.kc_top_k_name = prefix + "Online/LetterModel/top_k:0"
        self.key_length = prefix + "Online/LetterModel/batched_input_sequence_length:0"
        self.kc_state_in_name = prefix + "Online/LetterModel/state:0"
        self.kc_lm_state_in_name = prefix + "Online/LetterModel/lm_state_in:0"
        self.kc_input_name = prefix + "Online/LetterModel/batched_input_word_ids:0"
        self.kc_top_k_prediction_name = prefix + "Online/LetterModel/top_k_prediction:1"
        self.kc_output_name = prefix + "Online/LetterModel/probabilities:0"
        self.kc_state_out_name = prefix + "Online/LetterModel/state_out:0"

        with open(graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.per_process_gpu_memory_fraction = self._config.gpu_fraction
        self._sess = tf.Session(config=gpu_config)

    def predict(self, sentence, k):
        global probabilities, top_k_predictions, probability_topk, probability_p_topk, phrase_p_top_k
        inputs, inputs_key, word_letters = self._data_utility.sentence2ids(sentence)
        # print(inputs)
        # print(inputs_key)
        lm_state_out = np.zeros([self._config.num_layers, 2, 1, self._config.word_hidden_size], dtype=np.float32)
        kc_state_out = np.zeros([self._config.num_layers, 2, 1, self._config.letter_hidden_size], dtype=np.float32)
        words_out = list()
        phrase_logits = None
        if len(inputs) > 0:
            for i in range(len(inputs)):
                feed_values = {self.lm_input_name: [[inputs[i]]]}
                if i > 0:
                    feed_values[self.lm_state_in_name] = lm_state_out
                lm_state_out, phrase_p_top_k, phrase_p_prob, phrase_logits = self._sess.run([self.lm_state_out_name,
                                                                                               self.phrase_p_name,
                                                                                               self.phrase_p_probability,
                                                                                               self.phrase_logits],
                                                                                              feed_dict=feed_values)
                phrase_p_top_k = [id for id in phrase_p_top_k[0]]
                probability_p_topk = [phrase_p_prob[0][id] for id in phrase_p_top_k]

        for i in range(len(inputs_key)):
            feed_values = {self.kc_input_name: [[inputs_key[i]]],
                           self.kc_top_k_name: k}
            if i == 0 and len(inputs) > 0:
                feed_values[self.kc_lm_state_in_name] = lm_state_out
            else:
                feed_values[self.kc_state_in_name] = kc_state_out
            probabilities, top_k_predictions, kc_state_out = self._sess.run([self.kc_output_name, self.kc_top_k_prediction_name,
                                                                          self.kc_state_out_name], feed_dict=feed_values)
            probability_topk = [probabilities[0][id] for id in top_k_predictions[0]]
            words_out = self._data_utility.ids2outwords(top_k_predictions[0])
            if i == 0 and len(inputs) > 0:
                top_word = words_out[0]
                top_phrase = self._data_utility.get_top_phrase(phrase_logits, top_word)
                if top_phrase[0] is not None:
                    is_phrase_p, phrase_p = self.calculate_phrase_p(top_phrase, probability_p_topk, phrase_p_top_k)
                    words_out, probability_topk = self.final_words_out(words_out, top_phrase, phrase_p, probability_topk)

        return [{'word': word, 'probability': float(probability)}
                if word != '<unk>' else {'word': '<' + word_letters + '>', 'probability': float(probability)}
                for word, probability in zip(words_out, probability_topk)] if len(words_out) > 0 else []

    def predict_data(self, sentence, k):
        global probabilities, top_k_predictions, probability_topk, probability_p_topk, phrase_p_top_k
        sentence = sentence.rstrip()
        words_line, letters_line, words_ids, letters_ids, words_num, letters_num = self._data_utility.data2ids_line(sentence)
        out_str_list = []
        probability_topk_list = []
        # print(words_ids)
        # print(letters_ids)
        lm_state_out = np.zeros([self._config.num_layers, 2, 1, self._config.word_hidden_size], dtype=np.float32)
        kc_state_out = np.zeros([self._config.num_layers, 2, 1, self._config.letter_hidden_size], dtype=np.float32)

        for i in range(len(words_ids)):
            words_out = []
            probs_out = []
            feed_values = {self.lm_input_name: [[words_ids[i]]]}
            if i > 0:
                feed_values[self.lm_state_in_name] = lm_state_out

            lm_state_out, phrase_p_top_k, phrase_p_prob, phrase_logits = self._sess.run(
                [self.lm_state_out_name, self.phrase_p_name, self.phrase_p_probability,
                 self.phrase_logits], feed_dict=feed_values)
            phrase_p_top_k = [id for id in phrase_p_top_k[0]]
            probability_p_topk = [phrase_p_prob[0][id] for id in phrase_p_top_k]

            if i == len(letters_ids):
                break
            for j in range(len(letters_ids[i])):
                feed_values = {self.kc_input_name: [[letters_ids[i][j]]],
                               self.kc_top_k_name: k, self.key_length:[1]}

                if j == 0 and len(words_ids) > 0:
                    feed_values[self.kc_lm_state_in_name] = lm_state_out
                else:
                    feed_values[self.kc_state_in_name] = kc_state_out
                probabilities, top_k_predictions, kc_state_out = self._sess.run([self.kc_output_name, self.kc_top_k_prediction_name,
                                                                              self.kc_state_out_name], feed_dict=feed_values)
                probability_topk = [probabilities[0][id] for id in top_k_predictions[0]]
                words = self._data_utility.ids2outwords(top_k_predictions[0])

                if j == 0 and i > 0:
                    top_word = words[0]
                    top_phrase = self._data_utility.get_top_phrase(phrase_logits, top_word)
                    if top_phrase[0] is not None:
                        is_phrase_p, phrase_p = self.calculate_phrase_p(top_phrase, probability_p_topk, phrase_p_top_k)
                        words, probability_topk = self.final_words_out(words, top_phrase, phrase_p, probability_topk)
                words_out.append(words)
                probs_out.append(probability_topk)
            out_str = words_out if i > 0 else [['','','']] + words_out[1: ]
            out_str_list.append(out_str)
            probability_topk_list.append(probs_out)

        return words_line, letters_line, out_str_list, probability_topk_list

    def calculate_phrase_p(self, top_phrase, probability_p_topk, phrase_p_top_k):
        is_phrase_p = probability_p_topk[phrase_p_top_k.index(1)]
        phrase_p = is_phrase_p * top_phrase[1]
        return is_phrase_p, phrase_p

    def final_words_out(self, words, top_phrase, phrase_p, probability_topk):
        for i in range(len(probability_topk)):
            if phrase_p >= probability_topk[i]:
                probability_topk[i] = phrase_p
                words[i] = top_phrase[0]
                break
        return words, probability_topk

    def result_print(self, out_string, out_prob):
        string = ""
        for (word, prob) in zip(out_string, out_prob):
            prob = str(prob) if word != "" else "0.0"
            string = string + word + ":" + prob + "|"
        string = string[:-1]
        return string

    def predict_file(self, test_file_in, test_file_out, k):
        testfilein = open(test_file_in, "r")
        testfileout = open(test_file_out, 'w')
        t1 = time.time()
      
        for sentence in testfilein:
            sentence = sentence.rstrip()
            result = self.predict_data(sentence, k)

            if result is not None:
                words_line, letters_line, out_words_list, out_prob_list = result

                for i in range(len(out_words_list)):
                    print("\t".join(words_line[:i])
                         + "|#|" + letters_line[i]
                         + "|#|" + "\t".join(words_line[i:]) + "|#|"
                          + '\t'.join([self.result_print(out_words, out_prob)
                                       for (out_words, out_prob) in zip(out_words_list[i], out_prob_list[i])])
                          + "\n")
                    testfileout.write("\t".join(words_line[:i])
                                      + "|#|" + letters_line[i]
                                      + "|#|" + "\t".join(words_line[i:]) + "|#|"
                                      + '\t'.join([self.result_print(out_words, out_prob)
                                            for (out_words, out_prob) in zip(out_words_list[i], out_prob_list[i])])
                                      + "\n")

        t2 = time.time()
        print(t2 - t1)
        testfilein.close()
        testfileout.close()


if __name__ == "__main__":
    args = sys.argv

    graph_file = args[1]
    vocab_path = args[2]
    full_vocab = args[3]
    config_name = args[4]
    test_file_in = args[5]
    test_file_out = "test_result"
    engine = InputEngineRnn(graph_file, vocab_path, full_vocab, config_name)
    engine.predict_file(test_file_in, test_file_out, 3)

    # while True:
    #     sentence = input("please enter sentence:")
    #     if sentence == "quit()":
    #         exit()
    #     res = engine.predict(sentence, 10)
    #
    #     print(sentence)
    #     print(str(res))
