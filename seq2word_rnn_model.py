from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import config
import numpy as np
from data_feeder import data_iterator as data

FLAGS = config.FLAGS
data_type = config.data_type
index_data_type = config.index_data_type
np_index_data_type = config.np_index_data_type


class WordModel(object):
    """Static PTB model. Modified from old saniti-checked version of dynamic model.
    """

    def __init__(self, is_training, config):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.embedding_size = config.word_embedding_size
        self.hidden_size = config.word_hidden_size
        self.vocab_size_in = config.vocab_size_in
        self.vocab_size_out = config.vocab_size_out
        self.vocab_size_phrase = config.vocab_size_phrase
        self.input_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None], name="batched_input_word_ids")
        self.target_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None], name="batched_output_word_ids")
        self.output_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None], name="batched_output_word_masks")
        self.sequence_length = tf.placeholder_with_default(input=tf.fill(dims=[self.batch_size], value=self.num_steps),
                                                           shape=[self.batch_size], name="batched_input_sequence_length")
        self.top_k = tf.placeholder(dtype=index_data_type(), shape=[], name="top_k")

        # self.target_phrase_p = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
        #                                   name="batched_output_phrase_p_ids")
        # self.target_phrase_p_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
        #                                    name="batched_output_phrase_p_masks")
        # self.target_phrase_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
        #                                           name="batched_output_phrase_ids")
        # self.target_phrase_data_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
        #                                    name="batched_output_phrase_masks")
        # self.target_phrase_logits_masks = tf.placeholder_with_default(
        #     tf.ones([self.batch_size * self.num_steps, self.vocab_size_phrase], dtype=data_type()),
        #     [self.batch_size * self.num_steps, self.vocab_size_phrase], name="batched_output_phrase_logits_masks")
        # ####################################
        self.use_attention = True
        self.attention_size = 50
        ####################################

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        lstm_state_as_tensor_shape = [config.num_layers, 2, config.batch_size, config.word_hidden_size]#2表示c，h
        
        self._initial_state = tf.placeholder_with_default(tf.zeros(lstm_state_as_tensor_shape, dtype=data_type()),
                                                            lstm_state_as_tensor_shape, name="state")   #带有默认值的占位符，未获得输入前值为默认值

        unstack_state = tf.unstack(self._initial_state, axis=0)     #按行分解矩阵,即按照num_layers分解，分成num_layers个3维数组
        tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1])
             for idx in range(config.num_layers)]
        )

        #################################################################################################
        lstm_state_as_tensor_shape_1 = [config.num_layers, 2, 1, config.word_hidden_size]  # 2表示c，h

        self._initial_state_1 = tf.placeholder_with_default(tf.zeros(lstm_state_as_tensor_shape_1, dtype=data_type()),
                                                          lstm_state_as_tensor_shape_1,
                                                          name="state_1")  # 带有默认值的占位符，未获得输入前值为默认值

        unstack_state_1 = tf.unstack(self._initial_state_1, axis=0)  # 按行分解矩阵,即按照num_layers分解，分成num_layers个3维数组
        tuple_state_1 = tuple(
            [tf.contrib.rnn.LSTMStateTuple(unstack_state_1[idx][0], unstack_state_1[idx][1])
             for idx in range(config.num_layers)]
        )
        ################################################################################################

        # initial_state = cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope("Lm"):
            with tf.variable_scope("Embedding"):
                self._embedding = tf.get_variable("embedding", [self.vocab_size_in, self.embedding_size], dtype=data_type())
                inputs = tf.nn.embedding_lookup(self._embedding, self.input_data)
                embedding_to_rnn = tf.get_variable("embedding_to_rnn",
                                                  [self.embedding_size, self.hidden_size],
                                                  dtype=data_type())
                inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.embedding_size]),
                                          embedding_to_rnn),
                                shape=[self.batch_size, -1, self.hidden_size])

                if is_training and config.keep_prob < 1:
                    inputs = tf.nn.dropout(inputs, config.keep_prob)

            with tf.variable_scope("RNN"):
                outputs = list()
                states = list()
                state = tuple_state
                att_outputs = list()

                for timestep in range(self.num_steps):  #把几句话接起来，然后再按照num_steps分割

                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output, state) = cell(inputs[:, timestep, :], state)#output是(batch_size, hidden_size)的

                    outputs.append(output) #(t, B, D), t是当前timestep数

                    ######################
                    #####  Attention  ####
                    ######################
                    if self.use_attention:
                        rnn_output_t = tf.transpose(outputs, perm=[1, 0, 2]) #(B, t, D)
                        with tf.variable_scope('Attention_layer'):
                            w_attention = tf.get_variable('W_Attention', [self.hidden_size, self.attention_size],
                                                          dtype=data_type())
                            b_attention = tf.get_variable('b_Attention', [self.attention_size], dtype=data_type())
                            u_attention = tf.get_variable('u_Attention', [self.attention_size], dtype=data_type())
                            with tf.variable_scope('v'):
                                v = tf.tanh(tf.tensordot(rnn_output_t, w_attention, axes=1) + b_attention)
                            vu = tf.tensordot(v, u_attention, axes=1, name='vu')  # (B, t)
                            alphas = tf.nn.softmax(vu, name='alphas')  # (B, t)
                            attention_output = tf.reduce_sum(rnn_output_t * tf.expand_dims(alphas, -1), 1) #(B, D)
                            att_outputs.append(attention_output)
                    #####################

                    states.append(state)
                if self.use_attention:
                    rnn_output = tf.transpose(att_outputs, perm=[1, 0, 2]) #(B, T, D)
                else:
                    rnn_output = tf.transpose(outputs, perm=[1, 0, 2])   #最后是(B, T, D). 交换维度，交换第一第二维 ，因为append起来后，变成了(num_steps,batch_size,hidden_size),得变回(batchsize,num_steps,hidden_size)

                rnn_output = tf.reshape(rnn_output, [-1, self.hidden_size])
                states = tf.transpose(states, perm=[3, 1, 2, 0, 4])#同样得交换num_steps和batch_size
                unstack_states = tf.unstack(states, axis=0)#按照batch_size拆开成batchsize个array
                rnn_state = tf.concat(unstack_states, axis=2)#按照num_steps拼接,变成了[config.num_layers, 2, config.batch_size*num_steps, config.word_hidden_size]

                print("output shape:", rnn_output.shape)
                print("state shape:", rnn_state.shape)

            with tf.variable_scope("Softmax"):
                rnn_output_to_final_output = tf.get_variable("rnn_output_to_final_output",
                                                                    [self.hidden_size, self.embedding_size],#为了减少参数总数，做一步中间的维度转换
                                                                    dtype=data_type())
                self._softmax_w = tf.get_variable("softmax_w", [self.embedding_size, self.vocab_size_out],
                                                  dtype=data_type())
                softmax_b = tf.get_variable("softmax_b", [self.vocab_size_out], dtype=data_type())
        # with tf.variable_scope("PhraseProb"):#一个权重，判断词组有没有在词组表里，起一个降低phrase输出权重的作用。因为phrase的权重计算是部分交叉熵计算，所以有可能会很高，为了能与语言模型的输出一起进行排序，乘一个额外的权重将概率降下来。
        #     self._softmax_phrase_p_w = tf.get_variable("softmax_phrase_p_w", [self.embedding_size, 2],
        #                                                    dtype=data_type())
        #     softmax_phrase_p_b = tf.get_variable("softmax_phrase_p_b", [2], dtype=data_type())
        # with tf.variable_scope("Phrase"):
        #     self._softmax_phrase_w = tf.get_variable("softmax_phrase_w",
        #                                              [self.embedding_size, self.vocab_size_phrase],
        #                                              dtype=data_type())
        #     softmax_phrase_b = tf.get_variable("softmax_phrase_b", [self.vocab_size_phrase], dtype=data_type())
        #
        # self._logits_phrase_p = logits_phrase_p = tf.matmul(tf.matmul(rnn_output, rnn_output_to_final_output),
        #                                                             self._softmax_phrase_p_w) + softmax_phrase_p_b
        # self._phrase_p_probabilities = tf.nn.softmax(logits_phrase_p, name="phrase_p_probabilities")
        # _, phrase_p_prediction = tf.nn.top_k(logits_phrase_p, 2, name="phrase_p_prediction")   #返回值和索引，只取索引
        # logits_phrase = (tf.matmul(tf.matmul(rnn_output, rnn_output_to_final_output),
        #                            self._softmax_phrase_w) + softmax_phrase_b) * self.target_phrase_logits_masks#元素级别乘，rnnoutput的维度为(batchsize*numsteps, hiddensize)
        # self._logits_phrase = tf.identity(logits_phrase, name="logits_phrase")
        # self._phrase_probabilities = tf.nn.softmax(logits_phrase, name="phrase_probabilities")
        # _, self._phrase_prediction = tf.nn.top_k(logits_phrase, self.top_k, name="phrase_top_k_prediction")
        # loss_phrase_p = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits_phrase_p],
        #                                                                        [tf.reshape(self.target_phrase_p,
        #                                                                                    [-1])],
        #                                                                        [tf.reshape(
        #                                                                            self.target_phrase_p_masks,
        #                                                                            [-1])],
        #                                                                        average_across_timesteps=False)
        # loss_phrase = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits_phrase],
        #                                                                  [tf.reshape(self.target_phrase_data,
        #                                                                              [-1])],
        #                                                                  [tf.reshape(self.target_phrase_data_masks,
        #                                                                              [-1])],
        #                                                                  average_across_timesteps=False)
        # self._phrase_p_cost = phrase_p_cost = tf.reduce_sum(loss_phrase_p)
        #
        # self._phrase_cost = phrase_cost = tf.reduce_sum(loss_phrase)

        logits = tf.matmul(tf.matmul(rnn_output, rnn_output_to_final_output),
                            self._softmax_w) + softmax_b

        probabilities = tf.nn.softmax(logits, name="probabilities")
        _, top_k_prediction = tf.nn.top_k(logits, self.top_k, name="top_k_prediction")

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                  [tf.reshape(self.target_data, [-1])],
                                                                  [tf.reshape(self.output_masks, [-1])],
                                                                  average_across_timesteps=False)

        self._cost = cost = tf.reduce_sum(loss)

        self._final_state = tf.identity(state, "state_out")
        self._rnn_state = tf.identity(rnn_state, "rnn_state")
        self._logits = logits
        self._probabilities = probabilities
        self._top_k_prediction = top_k_prediction

        if not is_training:
            return

        self._lr = tf.get_variable(name="learning_rate", shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(config.learning_rate), trainable=False)
        tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/Lm")
        # tvars_phrase_p = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/PhraseProb")
        # tvars_phrase = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/Phrase")

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(0.001)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        # grads_phrase_p, _ = tf.clip_by_global_norm(tf.gradients(phrase_p_cost, tvars_phrase_p), config.max_grad_norm)
        # optimizer_phrase_p = tf.train.AdamOptimizer(0.001)
        # self._train_op_phrase_p = optimizer_phrase_p.apply_gradients(zip(grads_phrase_p, tvars_phrase_p),
        #                                                                      global_step=self.global_step)
        #
        # grads_phrase, _ = tf.clip_by_global_norm(tf.gradients(phrase_cost, tvars_phrase), config.max_grad_norm)
        # optimizer_phrase = tf.train.AdamOptimizer(0.001)
        # self._train_op_phrase = optimizer_phrase.apply_gradients(zip(grads_phrase, tvars_phrase),
        #                                                          global_step=self.global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def get_global_step(self, session):
        gs = session.run(self.global_step)
        return gs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def softmax_w(self):
        return self._softmax_w

    @property
    def cost(self):
        #return [self._cost, self._phrase_p_cost, self._phrase_cost]
        return [self._cost]

    @property
    def embedding(self):
        return self._embedding

    @property
    def final_state(self):
        return self._final_state

    @property
    def rnn_state(self):
        return self._rnn_state

    @property
    def lr(self):
        return self._lr

    @property
    def logits(self):
        #return [self._logits, self._logits_phrase_p, self._logits_phrase]
        return [self._logits]

    @property
    def probalities(self):
        #return [self._probabilities, self._phrase_p_probabilities,self._phrase_probabilities]
        return [self._probabilities]

    @property
    def top_k_prediction(self):
        #return [self._top_k_prediction, self._phrase_prediction]
        return [self._top_k_prediction]

    @property
    def train_op(self):
        #return [self._train_op, self._train_op_phrase_p, self._train_op_phrase]
        return [self._train_op]


class LetterModel(object):
    """Static PTB model. Modified from old saniti-checked version of dynamic model.
    """

    def __init__(self, is_training, config):
        self.num_steps = config.num_steps
        self.batch_size = config.batch_size * config.num_steps #语言模型中，一次输入，是一个batch_size*num_steps的矩阵，里面总共有batch_size*num_steps个单词。键码模型的样本是以单词为单位的。
        self.max_word_length = config.max_word_length
        self.embedding_size = config.letter_embedding_size
        self.hidden_size = config.letter_hidden_size
        self.vocab_size_in = config.vocab_size_letter
        self.vocab_size_out = config.vocab_size_out

        self.input_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                         name="batched_input_word_ids")
        self.target_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                          name="batched_output_word_ids")
        self.output_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                           name="batched_output_word_masks")
        self.sequence_length = tf.placeholder_with_default(input=tf.fill(dims=[self.batch_size], value=self.max_word_length),
                                                           shape=[self.batch_size],
                                                           name="batched_input_sequence_length")
        self.top_k = tf.placeholder(dtype=index_data_type(), shape=[], name="top_k")

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        lm_state_as_tensor_shape = [config.num_layers, 2, self.batch_size, config.word_hidden_size] #####################
        letter_state_as_tensor_shape = [config.num_layers, 2, self.batch_size, config.letter_hidden_size]

        self.lm_state_in = tf.placeholder_with_default(tf.zeros(lm_state_as_tensor_shape, dtype=data_type()),################
                                                          lm_state_as_tensor_shape, name="lm_state_in")
        with tf.variable_scope("StateMatrix"):

            lm_state_to_letter_state = tf.get_variable("lm_state_to_letter_state",
                                                       [config.word_hidden_size, config.letter_hidden_size],dtype=data_type())

        if config.word_hidden_size != config.letter_hidden_size:
            self._initial_state = tf.placeholder_with_default(
                                tf.reshape(tf.matmul(tf.reshape(self.lm_state_in, [-1, config.word_hidden_size]), lm_state_to_letter_state),
                                           letter_state_as_tensor_shape),
                                letter_state_as_tensor_shape, name="state")
        else:
            self._initial_state = tf.placeholder_with_default(
                self.lm_state_in, letter_state_as_tensor_shape, name="state")

        unstack_state = tf.unstack(self._initial_state, axis=0)
        tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1])
             for idx in range(config.num_layers)]
        )

        with tf.variable_scope("Embedding"):

            self._embedding = tf.get_variable("embedding", [self.vocab_size_in, self.embedding_size],
                                                  dtype=data_type())

            inputs = tf.nn.embedding_lookup(self._embedding, self.input_data)
            embedding_to_rnn = tf.get_variable("embedding_to_rnn",
                                               [self.embedding_size, self.hidden_size],
                                               dtype=data_type())
            inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.embedding_size]),
                                          embedding_to_rnn),
                                shape=[self.batch_size, -1, self.hidden_size])

            if is_training and config.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, config.keep_prob)

        with tf.variable_scope("RNN"):
            # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
            outputs, state_out = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.sequence_length,
                                                                 initial_state=tuple_state)

        output = tf.reshape(outputs, [-1, self.hidden_size])
        with tf.variable_scope("Softmax"):
            rnn_output_to_final_output = tf.get_variable("rnn_output_to_final_output",
                                                         [self.hidden_size, self.embedding_size],
                                                         dtype=data_type())
            self._softmax_w = tf.get_variable("softmax_w", [self.embedding_size, self.vocab_size_out],
                                              dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size_out], dtype=data_type())

        logits = tf.matmul(tf.matmul(output, rnn_output_to_final_output),
                           self._softmax_w) + softmax_b

        probabilities = tf.nn.softmax(logits, name="probabilities")
        _, top_k_prediction = tf.nn.top_k(logits, self.top_k, name="top_k_prediction")

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                  [tf.reshape(self.target_data, [-1])],
                                                                  [tf.reshape(self.output_masks, [-1])],
                                                                  average_across_timesteps=False)

        self._cost = cost = tf.reduce_sum(loss)

        self._final_state = tf.identity(state_out, "state_out")
        self._logits = logits
        self._probabilities = probabilities
        self._top_k_prediction = top_k_prediction

        if not is_training:
            return

        self._lr = tf.get_variable(name="learning_rate", shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(config.learning_rate), trainable=False)
        tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LetterModel")

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(0.001)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def get_global_step(self, session):
        gs = session.run(self.global_step)
        return gs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def softmax_w(self):
        return self._softmax_w

    @property
    def cost(self):
        return self._cost

    @property
    def embedding(self):
        return self._embedding

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def logits(self):
        return self._logits

    @property
    def probalities(self):
        return self._probabilities

    @property
    def top_k_prediction(self):
        return self._top_k_prediction

    @property
    def train_op(self):
        return self._train_op


