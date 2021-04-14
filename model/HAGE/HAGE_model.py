
import numpy as np
import tensorflow as tf


class HAGE_Model:
    def __init__(self, num_nodes, num_feat, feature_lens, num_categories, num_category_feat, category_feature_lens,
                 n_sampled=100, embedding_dim=128, lr=0.001):
        self.n_samped = n_sampled
        self.num_feat = num_feat
        self.feature_lens = feature_lens
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.lr = lr

        # parameters related to the second skip-gram
        self.num_category_feat = num_category_feat
        self.category_feature_lens = category_feature_lens
        self.category_embedding = self.category_embedding_init()

        self.softmax_w = tf.Variable(tf.truncated_normal((num_nodes, embedding_dim), stddev=0.1), name='softmax_w')
        self.softmax_b = tf.Variable(tf.zeros(num_nodes), name='softmax_b')
        self.inputs = self.input_init()
        self.embedding = self.embedding_init()
        self.alpha_embedding = tf.Variable(tf.random_uniform((num_nodes, num_feat+1), -1, 1))
        self.merge_emb = self.attention_merge()
        self.cost = self.make_skipgram_loss()
        # self.train_op = tf.train.AdagradOptimizer(lr).minimize(self.cost)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)

        # parameters related to the second skip-gram
        self.num_categories = num_categories
        self.category_softmax_w = tf.Variable(tf.truncated_normal((num_categories, embedding_dim), stddev=0.1),
                                              name='category_softmax_w')
        self.category_softmax_b = tf.Variable(tf.zeros(num_categories), name='category_softmax_b')
        self.item_softmax_w = tf.Variable(tf.truncated_normal((num_nodes, embedding_dim), stddev=0.1),
                                              name='item_softmax_w')
        self.item_softmax_b = tf.Variable(tf.zeros(num_nodes), name='item_softmax_b')
        self.category_category_inputs = self.category_input_init()
        self.category_item_inputs = self.category_input_init()
        # self.item_item_inputs = self.input_init()
        self.category_alpha_embedding = tf.Variable(tf.random_uniform((num_categories, num_category_feat), -1, 1))
        self.category_category_merge_emb = self.category_category_attention_merge()
        self.category_item_merge_emb = self.category_item_attention_merge()
        # self.item_item_merge_emb = self.item_item_attention_merge()
        self.category_category_cost = self.make_category_category_skipgram_loss()
        self.category_item_cost = self.make_category_item_skipgram_loss()
        # self.item_item_cost = self.make_item_item_skipgram_loss()
        self.category_category_train_op = tf.train.AdamOptimizer(lr).minimize(self.category_category_cost)
        self.category_item_train_op = tf.train.AdamOptimizer(lr).minimize(self.category_item_cost)
        # self.item_item_train_op = tf.train.AdamOptimizer(lr).minimize(self.item_item_cost)

    def embedding_init(self):
        cat_embedding_vars = []
        for i in range(self.num_feat):
            embedding_var = tf.Variable(tf.random_uniform((self.feature_lens[i], self.embedding_dim), -1, 1), name='embedding'+str(i),
                                        trainable=True)
            cat_embedding_vars.append(embedding_var)
        return cat_embedding_vars

    def attention_merge(self):
        embed_list = []
        num_embed_list = []
        for i in range(self.num_feat):
            cat_embed = tf.nn.embedding_lookup(self.embedding[i], self.inputs[i])
            embed_list.append(cat_embed)
        cat_embed = tf.nn.embedding_lookup(self.category_embedding[0], self.inputs[-2])
        embed_list.append(cat_embed)
        # TODO: add category attribute
        stack_embed = tf.stack(embed_list, axis=-1)
        # attention merge
        alpha_embed = tf.nn.embedding_lookup(self.alpha_embedding, self.inputs[0])
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        merge_emb = tf.reduce_sum(stack_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum
        return merge_emb

    def input_init(self):
        input_list = []
        for i in range(self.num_feat+1):
            input_col = tf.placeholder(tf.int32, [None], name='inputs_'+str(i))
            input_list.append(input_col)
        input_list.append(tf.placeholder(tf.int32, shape=[None, 1], name='label'))
        return input_list

    def make_skipgram_loss(self):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=self.softmax_w,
            biases=self.softmax_b,
            labels=self.inputs[-1],
            inputs=self.merge_emb,
            num_sampled=self.n_samped,
            num_classes=self.num_nodes,
            num_true=1,
            sampled_values=tf.random.uniform_candidate_sampler(
                true_classes=tf.cast(self.inputs[-1], tf.int64),
                num_true=1,
                num_sampled=self.n_samped,
                unique=True,
                range_max=self.num_nodes
            )
        ))
        return loss

    def category_input_init(self):
        input_list = []
        for i in range(self.num_category_feat):
            input_col = tf.placeholder(tf.int32, [None], name='category_inputs_'+str(i))
            input_list.append(input_col)
        input_list.append(tf.placeholder(tf.int32, shape=[None, 1], name='category_label'))
        return input_list

    def category_embedding_init(self):
        cat_embedding_vars = []
        for i in range(self.num_category_feat):
            embedding_var = tf.Variable(tf.random_uniform((self.category_feature_lens[i], self.embedding_dim), -1, 1),
                                        name='category_embedding' + str(i),
                                        trainable=True)
            cat_embedding_vars.append(embedding_var)
        return cat_embedding_vars

    def category_category_attention_merge(self):
        embed_list = []
        num_embed_list = []
        for i in range(self.num_category_feat):
            cat_embed = tf.nn.embedding_lookup(self.category_embedding[i], self.category_category_inputs[i])
            embed_list.append(cat_embed)
        stack_embed = tf.stack(embed_list, axis=-1)
        # attention merge
        alpha_embed = tf.nn.embedding_lookup(self.category_alpha_embedding, self.category_category_inputs[0])
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        merge_emb = tf.reduce_sum(stack_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum
        return merge_emb

    def make_category_category_skipgram_loss(self):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=self.category_softmax_w,
            biases=self.category_softmax_b,
            labels=self.category_category_inputs[-1],
            inputs=self.category_category_merge_emb,
            num_sampled=self.n_samped,
            num_classes=self.num_categories,
            num_true=1,
            sampled_values=tf.random.uniform_candidate_sampler(
                true_classes=tf.cast(self.category_category_inputs[-1], tf.int64),
                num_true=1,
                num_sampled=self.n_samped,
                unique=True,
                range_max=self.num_categories
            )
        ))
        return loss

    def category_item_attention_merge(self):
        embed_list = []
        num_embed_list = []
        for i in range(self.num_category_feat):
            cat_embed = tf.nn.embedding_lookup(self.category_embedding[i], self.category_item_inputs[i])
            embed_list.append(cat_embed)
        stack_embed = tf.stack(embed_list, axis=-1)
        # attention merge
        alpha_embed = tf.nn.embedding_lookup(self.category_alpha_embedding, self.category_item_inputs[0])
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        merge_emb = tf.reduce_sum(stack_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum
        return merge_emb

    def make_category_item_skipgram_loss(self):
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=self.softmax_w,
            biases=self.softmax_b,
            labels=self.category_item_inputs[-1],
            inputs=self.category_item_merge_emb,
            num_sampled=self.n_samped,
            num_classes=self.num_nodes,
            num_true=1,
            sampled_values=tf.random.uniform_candidate_sampler(
                true_classes=tf.cast(self.category_item_inputs[-1], tf.int64),
                num_true=1,
                num_sampled=self.n_samped,
                unique=True,
                range_max=self.num_nodes
            )
        ))
        return loss

    # def item_item_attention_merge(self):
    #     embed_list = []
    #     num_embed_list = []
    #     for i in range(self.num_feat):
    #         cat_embed = tf.nn.embedding_lookup(self.embedding[i], self.item_item_inputs[i])
    #         embed_list.append(cat_embed)
    #     # TODO: add category and category attribute
    #     stack_embed = tf.stack(embed_list, axis=-1)
    #     # attention merge
    #     alpha_embed = tf.nn.embedding_lookup(self.alpha_embedding, self.item_item_inputs[0])
    #     alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
    #     alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
    #     merge_emb = tf.reduce_sum(stack_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum
    #     return merge_emb
    #
    # def make_item_item_skipgram_loss(self):
    #     loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
    #         weights=self.softmax_w,
    #         biases=self.softmax_b,
    #         labels=self.item_item_inputs[-1],
    #         inputs=self.item_item_merge_emb,
    #         num_sampled=self.n_samped,
    #         num_classes=self.num_nodes,
    #         num_true=1,
    #         sampled_values=tf.random.uniform_candidate_sampler(
    #             true_classes=tf.cast(self.item_item_inputs[-1], tf.int64),
    #             num_true=1,
    #             num_sampled=self.n_samped,
    #             unique=True,
    #             range_max=self.num_nodes
    #         )
    #     ))
    #     return loss
