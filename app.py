from flask import Flask, render_template, request, url_for
from Bert.Bert_Model.Bert_Model import get_response
app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
        return render_template("index.html")

@app.route("/get")
def get_bot_response():
        userText = request.args.get("msg")
        print(get_response(userText))
        return get_response(userText)

if __name__ == '__main__':
        app.run(debug=True)
