from flask import Flask, redirect, url_for, request, render_template
import requests
import json
app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
        return render_template("index.html")

@app.route("/get")
def get_bot_response():
        userText = request.args.get("msg")
        data = json.dumps({"sender":"Rasa", "message":userText})
        headers = {'Content-type':'application/json', 'Accept':'text/plain'}
        res = requests.post('http://localhost:5005/webhooks/rest/webhook', data=data, headers=headers)
        res = res.json()
        userText = res[0]['text']
        return userText

if __name__ == '__main__':
        app.run(debug=True)
