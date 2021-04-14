from flask import Flask
from api import HAGE

app = Flask(__name__)
app.register_blueprint(HAGE.HAGE, url_prefix="/HAGE")

if __name__ == '__main__':
    # 启动APP
    app.run(host='0.0.0.0', port=5000, debug=False)
    # app.run(debug=True)
