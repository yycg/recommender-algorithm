from flask import Blueprint, request, Response
import json
import os
from gensim.models import KeyedVectors

HAGE = Blueprint('HAGE', __name__)


@HAGE.route("/recommend", methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        # 默认返回内容
        return_dict = {'returnCode': '200', 'returnInfo': '处理成功', 'result': False}
        # 判断入参是否为空
        if request.get_json() is None:
            return_dict['returnCode'] = '400'
            return_dict['returnInfo'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        # 获取传入的params参数
        get_data = request.get_json()
        law_titles = json.loads(get_data.get('lawTitles'))
        start = int(get_data.get('start'))
        count = int(get_data.get('count'))
        # 对参数进行操作
        return_dict['result'] = _recommend(law_titles, start, count)

        return Response(json.dumps(return_dict, ensure_ascii=False), mimetype='application/json')


def _recommend(law_titles, start, count):
    # load word vectors file
    wv = KeyedVectors.load_word2vec_format(os.path.abspath(os.path.dirname(__file__)).replace("\\", "/") +
                                           '/../data/HAGE.embed', binary=False)
    item_map = {}
    with open(os.path.abspath(os.path.dirname(__file__)).replace("\\", "/") + "/../data/item_map.csv", "r") as file:
        for line in file:
            id, code = line.strip().split('\t')
            item_map[id] = code

    item_score_list = []
    for cand in item_map.keys():
        score = sum([wv.similarity(str(item_map[cand]), str(item_map[item])) if item in item_map else 0 for item in law_titles])
        item_score_list.append((cand, score))
    item_score_list.sort(key=lambda item_score: item_score[1], reverse=True)

    return [item for item, _ in item_score_list[start:start+count]]


@HAGE.route("/hello", methods=['GET', 'POST'])
def hello():
    # 默认返回内容
    return_dict = {'return_code': '200', 'return_info': '处理成功', 'result': False}
    # 判断入参是否为空
    if request.args is None:
        return_dict['return_code'] = '400'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    # 获取传入的params参数
    get_data = request.args.to_dict()
    name = get_data.get('name')
    age = get_data.get('age')
    # 对参数进行操作
    return_dict['result'] = tt(name, age)

    return json.dumps(return_dict, ensure_ascii=False)


def tt(name, age):
    result_str = "%s今年%s岁" % (name, age)
    return result_str
