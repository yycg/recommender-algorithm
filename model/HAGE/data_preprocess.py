import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import contextlib
import time
import networkx as nx
import os
import itertools
import random
from deepwalk import graph
from model.HAGE.walker import RandomWalker
from util.database_utils import DatabaseUtils


class DataPreprocess(object):
    def __init__(self, p=0.25, q=2, num_walks=10, walk_length=10, window_size=5):
        db = DatabaseUtils()
        self.engine = db.get_db_engine()
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.window_size = window_size

    def accquire_data(self):
        """with illegal as (
            select * from t_discipline_illegal2_tf
            where STANDARD_ID in (
            SELECT ID FROM t_standard_word_tf
                WHERE IS_HISTORY = '0'
                  AND WORD_TYPE = '1'
                  AND ORG_CODE = '40083437-0'
            ))
        select * from illegal
        where DIS_ID in (
            select DIS_ID
            from illegal
            group by DIS_ID
            having count(1) > 1
        )
        order by DIS_ID;"""
        self.punish_illegal = pd.read_sql("""select * from (select * from t_discipline_illegal2_tf
                where STANDARD_ID in (
                    SELECT ID FROM t_standard_word_tf
                    WHERE IS_HISTORY = '0'
                    AND WORD_TYPE = '1'
                    AND ORG_CODE = '40083437-0'
                    AND TYPE_ID IN (SELECT ID FROM t_standard_word_type_tf)
                )) as illegal
            where DIS_ID in (
                select DIS_ID
                from (select * from t_discipline_illegal2_tf
                    where STANDARD_ID in (
                        SELECT ID FROM t_standard_word_tf
                        WHERE IS_HISTORY = '0'
                        AND WORD_TYPE = '1'
                        AND ORG_CODE = '40083437-0'
                        AND TYPE_ID IN (SELECT ID FROM t_standard_word_type_tf)
                    )) as illegal
                group by DIS_ID
                having count(1) > 1
            )
            order by DIS_ID;""", self.engine)
        print(self.punish_illegal)

        self.standard_word = pd.read_sql("""SELECT * FROM t_standard_word_tf
            WHERE IS_HISTORY = '0'
            AND WORD_TYPE = '1'
            AND ORG_CODE = '40083437-0'
            AND TYPE_ID IN (SELECT ID FROM t_standard_word_type_tf)""", self.engine)
        print(self.standard_word)

    def get_specialties(self):
        return pd.read_sql("""-- 专业类别 规范用语
            SELECT * FROM t_standard_word_name_tf
            WHERE IS_XZXK = '0' -- IS_XZXK 0 监督与处罚
            AND STATE = '0' -- STATE 0 启用
            AND TYPE = '0' -- TYPE 0 国家级
            ORDER BY SPECIALITY_CODE -- 按专业类别排序""", self.engine)

    def get_types(self, specialty_id, parent_id):
        return pd.read_sql("""-- 规范用语分类
            SELECT * FROM t_standard_word_type_tf
            WHERE IS_HISTORY = '0' -- 是否为历史 0正常 1历史
            AND WORD_TYPE = '1' -- 规范用语类别  1监督与处罚 0行政许可
            AND NAME_ID = '{0}' -- name表ID 对应专业类别
            AND PARENT_ID = '{1}' -- name表ID 对应专业类别
            AND ORG_CODE = '40083437-0' --  国家级规范用语
            ORDER BY LEVELS, SORT_ID -- 排序""".format(specialty_id, parent_id), self.engine)

    def get_standard_words(self, specialty_id, type_id):
        return pd.read_sql("""-- 规范用语表
            SELECT * FROM t_standard_word_tf
            WHERE IS_HISTORY = '0'
            AND WORD_TYPE = '1'
            AND NAME_ID = '{0}'
            AND ORG_CODE = '40083437-0'
            AND TYPE_ID = '{1}'
            ORDER BY SORT_ID""".format(specialty_id, type_id), self.engine)

    def traverse_specialties(self, category_item_children_map, category_category_children_map, category_map, item_map):
        category_map["root"] = len(category_map)

        for index, row in self.get_specialties().iterrows():
            id = row["ID"]
            category_map[id] = len(category_map)
            category_category_children_map.setdefault(category_map["root"], [])
            category_category_children_map[category_map["root"]].append(category_map[id])
            self.traverse_types(id, id, category_item_children_map, category_category_children_map, category_map, item_map)

    def traverse_types(self, name_id, parent_id, category_item_children_map, category_category_children_map,
                       category_map, item_map):
        for index, row in self.get_types(name_id, parent_id).iterrows():
            id = row["ID"]
            category_map[id] = len(category_map)
            category_category_children_map.setdefault(category_map[parent_id], [])
            category_category_children_map[category_map[parent_id]].append(category_map[id])
            if len(self.get_types(name_id, id)) > 0:
                self.traverse_types(name_id, id, category_item_children_map, category_category_children_map,
                                    category_map, item_map)
            else:
                self.traverse_standard_words(name_id, id, category_item_children_map, category_map, item_map)

    def traverse_standard_words(self, name_id, type_id, category_item_children_map, category_map, item_map):
        for index, row in self.get_standard_words(name_id, type_id).iterrows():
            id = row["ID"]
            item_map[id] = len(item_map)
            category_item_children_map.setdefault(category_map[type_id], [])
            category_item_children_map[category_map[type_id]].append(item_map[id])

    def preprocess_data(self):
        with elapsed_timer("-- {0}s - %s" % ("DFS tree, transform item ids",)):
            category_item_children_map = {}
            category_category_children_map = {}
            category_map = {}
            item_map = {}
            self.traverse_specialties(category_item_children_map, category_category_children_map, category_map, item_map)

            self.standard_word = self.standard_word[self.standard_word.ID.isin(item_map.keys())]
            self.punish_illegal = self.punish_illegal[self.punish_illegal.STANDARD_ID.isin(item_map.keys())]
            self.standard_word["ID"] = [item_map[id] for id in self.standard_word["ID"].tolist()]
            self.punish_illegal["STANDARD_ID"] = [item_map[id] for id in self.punish_illegal["STANDARD_ID"].tolist()]

            # sku_lbe = LabelEncoder()
            # # Fit label encoder and return encoded labels.
            # self.standard_word['ID'] = sku_lbe.fit_transform(self.standard_word['ID'])
            # # Transform labels to normalized encoding.
            # self.punish_illegal['STANDARD_ID'] = sku_lbe.transform(self.punish_illegal['STANDARD_ID'])

            with open(os.path.abspath('.').replace("\\", "/") + "/../../data/category_map.csv", "w") as file:
                for id, code in category_map.items():
                    file.write(id + "\t" + str(code) + "\n")

            with open(os.path.abspath('.').replace("\\", "/") + "/../../data/item_map.csv", "w") as file:
                for id, code in item_map.items():
                    file.write(id + "\t" + str(code) + "\n")

        with elapsed_timer("-- {0}s - %s" % ("make session list",)):
            case_items_map = {}
            for index, row in self.punish_illegal.iterrows():
                case = row["DIS_ID"]
                item = row["STANDARD_ID"]
                case_items_map.setdefault(case, set())
                case_items_map[case].add(item)
            session_list_all = []
            for _, items in case_items_map.items():
                session_list_all.append(list(items))

        with elapsed_timer("-- {0}s - %s" % ("session2graph",)):
            node_pair = dict()
            for session in session_list_all:
                for i in range(1, len(session)):
                    if (session[i - 1], session[i]) not in node_pair.keys():
                        node_pair[(session[i - 1], session[i])] = 1
                    else:
                        node_pair[(session[i - 1], session[i])] += 1

            in_node_list = list(map(lambda x: x[0], list(node_pair.keys())))
            out_node_list = list(map(lambda x: x[1], list(node_pair.keys())))
            weight_list = list(node_pair.values())
            graph_df = pd.DataFrame({'in_node': in_node_list, 'out_node': out_node_list, 'weight': weight_list})
            graph_df.to_csv(os.path.abspath('.').replace("\\", "/") + '/../../data/graph.csv', sep=' ', index=False, header=False)

        with elapsed_timer("-- {0}s - %s" % ("random walk",)):
            G = nx.read_edgelist(os.path.abspath('.').replace("\\", "/") + '/../../data/graph.csv', create_using=nx.DiGraph(), nodetype=None,
                                 data=[('weight', int)])
            walker = RandomWalker(G, p=self.p, q=self.q)
            print("Preprocess transition probs...")
            walker.preprocess_transition_probs()

            session_reproduce = walker.simulate_walks(num_walks=self.num_walks, walk_length=self.walk_length, workers=4,
                                                      verbose=1)
            session_reproduce = list(filter(lambda x: len(x) > 2, session_reproduce))

        with elapsed_timer("-- {0}s - %s" % ("add side info",)):
            # df = self.standard_word
            # product_data = df.loc[:, ["ID"]]
            # product_data = product_data.rename(columns={'ID': 'sku_id'})
            #
            # all_skus = self.standard_word['ID'].unique()
            # all_skus = pd.DataFrame({'sku_id': list(all_skus)})
            #
            # # Transform labels back to original encoding.
            # all_skus['sku_id'] = sku_lbe.inverse_transform(all_skus['sku_id'])
            # print("sku nums: " + str(all_skus.count()))
            # sku_side_info = pd.merge(all_skus, product_data, on='sku_id', how='left').fillna("NaN")
            #
            # # id2index
            # for feat in sku_side_info.columns:
            #     if feat != 'sku_id':
            #         lbe = LabelEncoder()
            #         sku_side_info[feat] = lbe.fit_transform(sku_side_info[feat])
            #     else:
            #         sku_side_info[feat] = sku_lbe.transform(sku_side_info[feat])
            #
            # sku_side_info = sku_side_info.sort_values(by=['sku_id'], ascending=True)
            # sku_side_info.to_csv('../../../data/sku_side_info.csv', index=False, header=False, sep='\t')

            standard_word = self.standard_word.loc[:, ["ID"]]
            standard_word = standard_word.rename(columns={'ID': 'sku_id'})

            standard_word.to_csv(os.path.abspath('.').replace("\\", "/") + '/../../data/sku_side_info.csv', index=False, header=False, sep='\t')

        with elapsed_timer("-- {0}s - %s" % ("get pair",)):
            all_pairs = get_graph_context_all_pairs(session_reproduce, self.window_size)
            np.savetxt(os.path.abspath('.').replace("\\", "/") + '/../../data/all_pairs', X=all_pairs, fmt="%d", delimiter=" ")

        with elapsed_timer("-- {0}s - %s" % ("add category",)):
            standard_word = self.standard_word.loc[:, ["ID", "TYPE_ID"]]
            categories = []
            for index, row in standard_word.iterrows():
                categories.append(category_map[row["TYPE_ID"]])
            standard_word["TYPE_ID"] = categories
            standard_word = standard_word.rename(columns={'ID': 'sku_id', 'TYPE_ID': 'Category'})
            standard_word.to_csv(os.path.abspath('.').replace("\\", "/") + '/../../data/sku_side_info_category.csv', index=False, header=False, sep='\t')

            with open(os.path.abspath('.').replace("\\", "/") + "/../../data/category_category_children.csv", "w") as file:
                for category, category_children in category_category_children_map.items():
                    file.write(
                        str(category) + "\t" + ",".join([str(category) for category in category_children]) + "\n")

            with open(os.path.abspath('.').replace("\\", "/") + "/../../data/category_item_children.csv", "w") as file:
                for category, item_children in category_item_children_map.items():
                    file.write(str(category) + "\t" + ",".join([str(item) for item in item_children]) + "\n")

            with open(os.path.abspath('.').replace("\\", "/") + "/../../data/category_side_info.csv", "w", encoding="utf-8") as file:
                for i in range(len(category_map)):
                    file.write(str(i) + "\n")

        with elapsed_timer("-- {0}s - %s" % ("build graph",)):
            with open(os.path.abspath('.').replace("\\", "/") + "/../../data/category_category_children.csv", "r") as reader, \
                    open(os.path.abspath('.').replace("\\", "/") + "/../../data/category_category_children_edge.txt", "w") as writer:
                for line in reader:
                    columns = line.strip().split("\t")
                    category = columns[0]
                    items = columns[1].split(",")
                    edges = itertools.permutations(items, 2)
                    for edge in edges:
                        writer.write(category + " " + " ".join(edge) + "\n")

            with open(os.path.abspath('.').replace("\\", "/") + "/../../data/category_item_children.csv", "r") as reader, \
                    open(os.path.abspath('.').replace("\\", "/") + "/../../data/category_item_children_edge.txt", "w") as writer:
                for line in reader:
                    columns = line.strip().split("\t")
                    category = columns[0]
                    items = columns[1].split(",")
                    edges = itertools.permutations(items, 2)
                    for edge in edges:
                        writer.write(category + " " + " ".join(edge) + "\n")

        with elapsed_timer("-- {0}s - %s" % ("random walk",)):
            category_category_children_graph_map = load_category_edgelist(
                os.path.abspath('.').replace("\\", "/") + "/../../data/category_category_children_edge.txt")
            category_category_children_walks = random_walk(category_category_children_graph_map, self.num_walks,
                                                           self.walk_length)

            category_item_children_graph_map = load_category_edgelist(
                os.path.abspath('.').replace("\\", "/") + "/../../data/category_item_children_edge.txt")
            category_item_children_walks = random_walk(category_item_children_graph_map, self.num_walks,
                                                       self.walk_length)

        with elapsed_timer("-- {0}s - %s" % ("get pair",)):
            num_items = len(self.standard_word['ID'])
            category_category_all_pairs, category_item_all_pairs, item_item_all_pairs = \
                get_category_graph_context_all_pairs(category_category_children_walks,
                                                     category_item_children_walks,
                                                     self.window_size, num_items)
            np.savetxt(os.path.abspath('.').replace("\\", "/") + '/../../data/category_category_all_pairs', X=category_category_all_pairs,
                       fmt="%d", delimiter=" ")
            np.savetxt(os.path.abspath('.').replace("\\", "/") + '/../../data/category_item_all_pairs', X=category_item_all_pairs, fmt="%d",
                       delimiter=" ")
            np.savetxt(os.path.abspath('.').replace("\\", "/") + '/../../data/item_item_all_pairs', X=item_item_all_pairs, fmt="%d", delimiter=" ")


class StubLogger(object):
    def __getattr__(self, name):
        return self.log_print

    def log_print(self, msg, *args):
        print(msg % args)


LOGGER = StubLogger()
LOGGER.info("Hello %s!", "world")


@contextlib.contextmanager
def elapsed_timer(message):
    start_time = time.time()
    yield
    LOGGER.info(message.format(time.time() - start_time))


def get_graph_context_all_pairs(walks, window_size):
    all_pairs = []
    for k in range(len(walks)):
        for i in range(len(walks[k])):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 0 or j >= len(walks[k]):
                    continue
                else:
                    all_pairs.append([walks[k][i], walks[k][j]])
    return np.array(all_pairs, dtype=np.int32)


def load_category_edgelist(file_, undirected=True):
    category_graph_map = {}
    with open(file_) as f:
        for l in f:
            c, x, y = l.strip().split()[:3]
            x = int(x)
            y = int(y)
            category_graph_map.setdefault(c, graph.Graph())
            G = category_graph_map[c]
            G[x].append(y)
            if undirected:
                G[y].append(x)

    for category, G in category_graph_map.items():
        G.make_consistent()

    return category_graph_map


def random_walk(category_graph_map, num_paths, path_length, alpha=0, rand=random.Random(0)):
    for category, G in category_graph_map.items():
        nodes = list(G.nodes())

        for cnt in range(num_paths):
            rand.shuffle(nodes)
            for node in nodes:
                walk = G.random_walk(path_length, rand=rand, alpha=alpha, start=node)
                yield [category] + walk


def get_category_graph_context_all_pairs(category_category_children_walks, category_item_children_walks,
                                         window_size, num_items):
    category_category_all_pairs = []
    category_item_all_pairs = []
    item_item_all_pairs = []

    # category_category_children_walks
    for walk in category_category_children_walks:
        for i in range(len(walk)):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 1 or j >= len(walk):
                    continue
                else:
                    # (category, category)
                    category_category_all_pairs.append([int(walk[i]), int(walk[j])])

    # category_item_children_walks
    for walk in category_item_children_walks:
        for i in range(len(walk)):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 1 or j >= len(walk):
                    continue
                elif i == 0:
                    # (category, item)
                    category_item_all_pairs.append([int(walk[i]), int(walk[j])])
                else:
                    # (item, item)
                    item_item_all_pairs.append([int(walk[i]), int(walk[j])])

    return (np.array(category_category_all_pairs, dtype=np.int32),
            np.array(category_item_all_pairs, dtype=np.int32),
            np.array(item_item_all_pairs, dtype=np.int32))


if __name__ == '__main__':
    dataPreprocess = DataPreprocess()
    dataPreprocess.accquire_data()
    dataPreprocess.preprocess_data()
