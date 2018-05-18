#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import numpy as np
from collections import defaultdict


class DataUtility:
    def __init__(self, vocab_file_in_words=None, vocab_file_in_letters=None, vocab_file_out=None,
                 vocab_file_phrase=None, full_vocab_file_in_words=None):

        self.start_str = "<start>"
        self.eos_str = "<eos>"
        self.unk_str = "<unk>"
        self.num_str = "<num>"
        self.pun_str = "<pun>"
        self.unk_str2 = "<und>"
        self.fullvocab_set = None
        self.pad_id = 0

        if vocab_file_in_words and vocab_file_in_letters and vocab_file_out:
            self.id2token_in_words, self.id2token_in_letters, self.id2token_out, self.id2token_phrase = {}, {}, {}, {}
            self.token2id_in_words, self.token2id_in_letters, self.token2id_out, self.token2id_phrase = {}, {}, {}, {}
            with open(vocab_file_in_words, mode="r") as f:#vocab_in_words
                for line in f:
                    token, id = line.strip().split("##")#以##分割开，前面是词，后面是id
                    id = int(id)
                    self.id2token_in_words[id] = token
                    self.token2id_in_words[token] = id
            self.in_words_count = len(self.token2id_in_words)#一共几个不同的词
            self.eos_id = self.token2id_in_words[self.eos_str]#EOS的id

            with open(vocab_file_in_letters, mode="r") as f:#vocab_in_letters
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_in_letters[id] = token
                    self.token2id_in_letters[token] = id
            self.start_id = self.token2id_in_letters[self.start_str]
            self.in_letters_count = len(self.token2id_in_letters)

            with open(vocab_file_out, mode="r") as f:#vocab_out
                for line in f:
                    token, id = line.split("##")
                    id = int(id)
                    self.id2token_out[id] = token
                    self.token2id_out[token] = id
            self.out_words_count = len(self.token2id_out)

            with open(vocab_file_phrase, mode="r") as f:#vocab_phrase
                for line in f:
                    token, id = line.split("##")
                    id = int(id)
                    self.id2token_phrase[id] = token
                    self.token2id_phrase[token] = id
            self.phrase_count = len(self.token2id_phrase)#一共多少个词组

        if full_vocab_file_in_words:
            self.fullvocab_set = set()
            with open(full_vocab_file_in_words, "r") as f:
                for line in f:
                    token, freq = line.split()
                    self.fullvocab_set.add(token)
            sys.stderr.write("Full vocabulary size: %d " % len(self.fullvocab_set))

        self.head_mask = defaultdict(lambda: np.zeros(shape=(self.phrase_count,), dtype=np.float32))#默认值为长度为词组数量的0向量
        self.head_pick_id = None

        for phrase, phrase_id in self.token2id_phrase.items():#遍历词组与词组对应的id
            if phrase.split()[0] in self.token2id_out:#词组的第一个单词在out字典里
                head_id_of_phrase = self.token2id_out[phrase.split()[0]]#取出词组第一个单词的out_id
                self.head_mask[head_id_of_phrase][phrase_id] = 1.0#第一个单词是out_id对应单词的词组，在向量中对应词组id的位置设为1
        for _, word_id in self.token2id_out.items():#遍历out字典里的id
            if word_id in self.head_mask:
                continue
            self.head_mask[word_id][0] = 0

    def get_head_pick_id(self, head):
        if head is None:
            return None
        if self.head_pick_id is None:
            self.head_pick_id = defaultdict(list)
            for phrase, phrase_id in self.token2id_phrase.items():
                self.head_pick_id[phrase.split()[0]].append(phrase_id)
        return None if head not in self.head_pick_id else self.head_pick_id[head]

    def get_top_phrase(self, logits, head):
        pick_id = self.get_head_pick_id(head)
        if pick_id is None:
            return None, None
        logits_on_head = np.take(logits, pick_id)
        p_on_head = self.softmax(logits_on_head)
        top_id = np.argmax(p_on_head)
        top_phrase = self.id2token_phrase[pick_id[top_id]]
        top_p = p_on_head[top_id]
        return top_phrase, top_p

    def softmax(self, logits):
        exp_logits = np.exp(logits)
        exp_sum = np.expand_dims(np.sum(exp_logits, -1), -1)
        return exp_logits / exp_sum

    def word2id(self, word):
        if re.match("^[a-zA-Z]$", word) or (word in self.token2id_in_words):
            word_out = word
        else:
            if re.match("^[+-]*[0-9]+.*[0-9]*$", word):
                word_out = self.num_str
            else:
                if re.match("^[^a-zA-Z0-9']*$", word):
                    word_out = self.pun_str
                else:
                    word_out = self.unk_str
        rid = self.token2id_in_words.get(word_out, -1)
        if rid == -1:
            if self.fullvocab_set and word_out in self.fullvocab_set:
                return self.token2id_in_words[self.unk_str2]
            else:
                return self.token2id_in_words[self.unk_str]
        return rid

    def words2ids(self, words):
        # words_split = re.split("\\s+", words)
        return [self.eos_id] + [self.word2id(word) for word in words if len(word) > 0]

    def letters2ids(self, letters):
        letters_split = re.split("\\s+", letters)
        return [self.start_id] + [self.token2id_in_letters.get(letter, self.token2id_in_letters[self.unk_str])
                                  for letter in letters_split if len(letter) > 0]

    def outword2id(self, outword):
        return self.token2id_out.get(outword, self.token2id_out[self.unk_str])

    def ids2outwords(self, ids_out):
        return [self.id2token_out.get(id, self.unk_str) for id in ids_out]

    def ids2inwords(self, ids_in):
        return [self.id2token_in_words.get(int(id), self.unk_str) for id in ids_in]

    def ids2phrase(self, ids_phrase):
        return [self.id2token_phrase.get(id, self.unk_str) for id in ids_phrase]

    def phrase2id(self, phrases):
        return [self.token2id_phrase.get(phrase, 0) for phrase in phrases]

    def data2ids_line(self, data_line):
        data_line_split = re.split("\\|#\\|", data_line)
        letters_line = data_line_split[0].split("\t")
        words_line = data_line_split[1].strip().split("\t")
        words_ids = self.words2ids(words_line)
        letters_ids = [self.letters2ids(letters) for letters in letters_line]
        words_num = len(words_ids)
        letters_num = [len(letter_ids) for letter_ids in letters_ids]
        return words_line, letters_line, words_ids, letters_ids, words_num, letters_num

    def sentence2ids(self, sentence):
        words_array = re.split('\\s+', sentence)
        word_letters = words_array[-1]
        words_array = words_array[:-1]
        # words = ' '.join(words_array)
        letters = ' '.join(word_letters)
        words_ids = self.words2ids(words_array)
        letters_ids = self.letters2ids(letters)
        return words_ids, letters_ids, word_letters
