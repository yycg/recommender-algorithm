
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import argparse
import os
from model.HAGE.HAGE_model import HAGE_Model
from model.HAGE.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--batch_size", type=int, default=2048)
    # 采样次数
    parser.add_argument("--n_sampled", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--root_path", type=str, default=os.path.abspath('.').replace("\\", "/") + '/../../data/')
    # sku_side_info.csv的列数，第一列是物品id，后面列是特征id
    parser.add_argument("--num_feat", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--outputEmbedFile", type=str, default=os.path.abspath('.').replace("\\", "/") +
                                                               '/../../data/HAGE.embed')
    # parser.add_argument("--outputCategoryEmbedFile", type=str, default='../../../data/amazon/embedding/category.embed')
    # parameters related to the second skip-gram
    parser.add_argument("--num_category_feat", type=int, default=1)
    args = parser.parse_args()

    # read train_data
    print('read features...')
    start_time = time.time()
    side_info = np.loadtxt(args.root_path + 'sku_side_info_category.csv', dtype=np.int32, delimiter='\t')
    all_pairs = np.loadtxt(args.root_path + 'all_pairs', dtype=np.int32, delimiter=' ')
    # feature_lens表示side_info中每个feature的取值个数，例如feature_lens = [3933, 1692]
    feature_lens = []
    # side_info = side_info.reshape((len(side_info), 1))
    for i in range(side_info.shape[1]):
        tmp_len = len(set(side_info[:, i]))
        feature_lens.append(tmp_len)
    end_time = time.time()
    print('time consumed for read features: %.2f' % (end_time - start_time))

    # read train data of the second skip-gram
    print('read features of the second skip-gram...')
    start_time = time.time()
    category_side_info = np.loadtxt(args.root_path + 'category_side_info.csv', dtype=np.int32, delimiter='\t')
    category_category_all_pairs = np.loadtxt(args.root_path + 'category_category_all_pairs', dtype=np.int32,
                                             delimiter=' ')
    category_item_all_pairs = np.loadtxt(args.root_path + 'category_item_all_pairs', dtype=np.int32, delimiter=' ')
    item_item_all_pairs = np.loadtxt(args.root_path + 'item_item_all_pairs', dtype=np.int32, delimiter=' ')
    # feature_lens表示side_info中每个feature的取值个数，例如feature_lens = [3933, 1692]
    category_feature_lens = []
    category_side_info = category_side_info.reshape((len(category_side_info), 1))
    for i in range(category_side_info.shape[1]):
        tmp_len = len(set(category_side_info[:, i]))
        category_feature_lens.append(tmp_len)
    end_time = time.time()
    print('time consumed for read features of the second skip-gram: %.2f' % (end_time - start_time))

    HAGE = HAGE_Model(len(side_info), args.num_feat, feature_lens, len(category_side_info), args.num_category_feat,
                      category_feature_lens, n_sampled=args.n_sampled, embedding_dim=args.embedding_dim, lr=args.lr)

    # init model
    print('init...')
    start_time = time.time()
    init = tf.global_variables_initializer()
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    sess = tf.Session(config=config_tf)
    sess.run(init)
    end_time = time.time()
    print('time consumed for init: %.2f' % (end_time - start_time))

    print_every_k_iterations = 100
    loss = 0
    category_category_loss = 0
    category_item_loss = 0
    item_item_loss = 0
    iteration = 0
    start = time.time()

    max_iter = len(all_pairs)//args.batch_size*args.epochs
    category_category_batch_size = len(category_category_all_pairs)*args.epochs//max_iter
    category_item_batch_size = len(category_item_all_pairs)*args.epochs//max_iter
    item_item_batch_size = len(item_item_all_pairs) * args.epochs // max_iter
    for iter in range(max_iter):
        iteration += 1
        batch_features, batch_labels = next(graph_context_batch_iter(all_pairs, args.batch_size, side_info,
                                                                     args.num_feat))
        feed_dict = {input_col: batch_features[:, i] for i, input_col in enumerate(HAGE.inputs[:-1])}
        feed_dict[HAGE.inputs[-1]] = batch_labels
        _, train_loss = sess.run([HAGE.train_op, HAGE.cost], feed_dict=feed_dict)

        loss += train_loss

        if iteration % print_every_k_iterations == 0:
            end = time.time()
            e = iteration*args.batch_size//len(all_pairs)
            print("Epoch {}/{}".format(e, args.epochs),
                  "Iteration: {}".format(iteration),
                  "Avg. Training loss: {:.4f}".format(loss / print_every_k_iterations),
                  "{:.4f} sec/batch".format((end - start) / print_every_k_iterations))
            loss = 0
            start = time.time()

        # train the second skip-gram
        # (category, category)
        batch_features, batch_labels = next(
            graph_context_batch_iter(category_category_all_pairs, category_category_batch_size,
                                     category_side_info, args.num_category_feat))
        feed_dict = {input_col: batch_features[:, i] for i, input_col in enumerate(HAGE.category_category_inputs[:-1])}
        feed_dict[HAGE.category_category_inputs[-1]] = batch_labels
        _, train_loss = sess.run([HAGE.category_category_train_op, HAGE.category_category_cost], feed_dict=feed_dict)

        category_category_loss += train_loss

        if iteration % print_every_k_iterations == 0:
            end = time.time()
            e = iteration * category_category_batch_size // len(category_category_all_pairs)
            print("Epoch {}/{}".format(e, args.epochs),
                  "Iteration: {}".format(iteration),
                  "Avg. Training loss: {:.4f}".format(category_category_loss / print_every_k_iterations),
                  "{:.4f} sec/batch".format((end - start) / print_every_k_iterations))
            category_loss = 0
            start = time.time()

        # (category, item)
        batch_features, batch_labels = next(
            graph_context_batch_iter(category_item_all_pairs, category_item_batch_size,
                                     category_side_info, args.num_category_feat))
        feed_dict = {input_col: batch_features[:, i] for i, input_col in enumerate(HAGE.category_item_inputs[:-1])}
        feed_dict[HAGE.category_item_inputs[-1]] = batch_labels
        _, train_loss = sess.run([HAGE.category_item_train_op, HAGE.category_item_cost], feed_dict=feed_dict)

        category_item_loss += train_loss

        if iteration % print_every_k_iterations == 0:
            end = time.time()
            e = iteration * category_item_batch_size // len(category_item_all_pairs)
            print("Epoch {}/{}".format(e, args.epochs),
                  "Iteration: {}".format(iteration),
                  "Avg. Training loss: {:.4f}".format(category_item_loss / print_every_k_iterations),
                  "{:.4f} sec/batch".format((end - start) / print_every_k_iterations))
            category_loss = 0
            start = time.time()

        # (item, item)
        # batch_features, batch_labels = next(
        #     graph_context_batch_iter(item_item_all_pairs, item_item_batch_size, side_info, args.num_feat))
        # feed_dict = {input_col: batch_features[:, i] for i, input_col in enumerate(HAGE.item_item_inputs[:-1])}
        # feed_dict[HAGE.item_item_inputs[-1]] = batch_labels
        # _, train_loss = sess.run([HAGE.item_item_train_op, HAGE.item_item_cost], feed_dict=feed_dict)
        #
        # item_item_loss += train_loss
        #
        # if iteration % print_every_k_iterations == 0:
        #     end = time.time()
        #     e = iteration * item_item_batch_size // len(item_item_all_pairs)
        #     print("Epoch {}/{}".format(e, args.epochs),
        #           "Iteration: {}".format(iteration),
        #           "Avg. Training loss: {:.4f}".format(item_item_loss / print_every_k_iterations),
        #           "{:.4f} sec/batch".format((end - start) / print_every_k_iterations))
        #     category_loss = 0
        #     start = time.time()

    print('optimization finished...')
    saver = tf.train.Saver()
    saver.save(sess, "checkpoints/HAGE")


    feed_dict_test = {input_col: list(side_info[:, i]) for i, input_col in enumerate(HAGE.inputs[:-1])}
    feed_dict_test[HAGE.inputs[-1]] = np.zeros((len(side_info), 1), dtype=np.int32)
    embedding_result = sess.run(HAGE.merge_emb, feed_dict=feed_dict_test)
    print('saving embedding result...')
    write_embedding(embedding_result, args.outputEmbedFile, args.embedding_dim)

    # feed_dict_test = {input_col: list(category_side_info[:, i])
    #                   for i, input_col in enumerate(HAGE.category_category_inputs[:-1])}
    # feed_dict_test[HAGE.category_category_inputs[-1]] = np.zeros((len(category_side_info), 1), dtype=np.int32)
    # embedding_result = sess.run(HAGE.category_category_merge_emb, feed_dict=feed_dict_test)
    # print('saving category embedding result...')
    # write_embedding(embedding_result, args.outputEmbedFile, args.embedding_dim)

    # print('visualization...')
    # plot_embeddings(embedding_result[:5000, :], side_info[:5000, :])


