from flask import Flask, render_template,request,url_for

#chatterbot dependencies
from chatterbot import ChatBot
import webbrowser
from chatterbot.trainers import ChatterBotCorpusTrainer
BOTNAME = "Simon"
app = Flask(__name__)
english_bot = ChatBot(BOTNAME, 
        logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand.',
            'maximum_similarity_threshold': 0.90,
        },        
    ],
		preprocessors = [
			"chatterbot.preprocessors.clean_whitespace",
		],
		input_adaptor="chatterbot.input.TerminalAdaptor",
        output_adaptor="chatterbot.output.TerminalAdaptor",
		database_uri='sqlite:///database.sqlite3')
trainer = ChatterBotCorpusTrainer(english_bot)

	# Train based on the english corpus
trainer.train(
		"chatterbot.corpus.english",
		"chatterbot.corpus.english.greetings",
		"chatterbot.corpus.english.conversations",
		)
#trainer = ChatterBotCorpusTrainer(english_bot)
#trainer.train("chatterbot.corpus.english")
#trainer.train("data/data.yml")
print(f"Hello I am {BOTNAME}")


@app.route("/")
def index():
    return render_template("index.html")



    
    
 
@app.route("/get")


def get_bot_response():
    
    userText=request.args.get("msg") #get data from input, we write js to index.html
    
    
    if userText=="Bye" or request=='bye':
        return('Bye')
        
    elif userText=="PPP":
        webbrowser.open('http://localhost:8888/notebooks/2.%20%20Extras_BDA/LGM/Emotion%20based%20song%20recomendation.ipynb')
        return(" You will be taken to the link, hold on")
        #return(webbrowser.open('https://www.google.com/search?q=ant&rlz=1C1UEAD_enIN964IN964&oq=ant&aqs=chrome..69i57j46i131i433i512j46i433i512j0i433i512j0i512j69i65l2j69i61.607j0j7&sourceid=chrome&ie=UTF-8'))
        
    else:
        #userText=english_bot.get_response(request)
        #print('Bot: ', response)
        #bot_input = input("You: ")
        #bot_respose = bot.get_response(bot_input)
        #print(f"{BOTNAME}: {bot_respose}")


        return str(english_bot.get_response(userText))

if __name__ == "__main__":
    app.run(debug=True)
