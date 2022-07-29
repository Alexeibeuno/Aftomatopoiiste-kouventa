Step1: Install rasa

Step2: You have the flask code on the app.py. In the requests.post, you'll find a link. That's used to connect the rasa model that's being trained.

Step3: Before running the flask code, in a seperate terminal, run:
Code: rasa run -m models --enable-api
