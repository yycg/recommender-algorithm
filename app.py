from flask import Flask
from api import item2vec

app = Flask(__name__)
app.register_blueprint(item2vec.recommend, url_prefix="/item2vec")

if __name__ == '__main__':
    # 启动APP
    app.run(host='0.0.0.0', port=5000, debug=False)
    # app.run(debug=True)
