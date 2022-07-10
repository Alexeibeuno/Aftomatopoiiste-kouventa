import json

from chatterbot import ChatBot
from chatterbot.trainers import UbuntuCorpusTrainer

# Create a new chat bot named Charlie
chatbot = ChatBot('Charlie')


trainer = UbuntuCorpusTrainer(chatbot)

trainer.train()

# Get a response to the input text 'I would like to book a flight.'
while True:
    request=input('You:')
    
    if request=="Bye" or request=='bye':
        print('Bot: Bye')
        break
    if request=="pLay_songs":
        open('http://localhost:8888/notebooks/2.%20%20Extras_BDA/LGM/Emotion%20based%20song%20recomendation.ipynb')
        break
    else:
        response=chatbot.get_response(request)
        print('Bot: ', response)
    

print(response)
