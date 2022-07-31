from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import webbrowser
BOTNAME = "Simon"

def start():
	bot = ChatBot(BOTNAME,
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

	trainer = ChatterBotCorpusTrainer(bot)

	# Train based on the english corpus
	trainer.train(
		"chatterbot.corpus.english",
		"chatterbot.corpus.english.greetings",
		"chatterbot.corpus.english.conversations",
		)


	print(f"Hello I am {BOTNAME}")

	while True:
		try:
			request=input('You:')
    
			if request=="Bye" or request=='bye':
				print('Bot: Bye')
				break
			if request=="PPP":
				print("Bot: You will be taken to the link, hold on")
				webbrowser.open('http://localhost:8888/notebooks/2.%20%20Extras_BDA/LGM/Emotion%20based%20song%20recomendation.ipynb')
				break
			else:
				response=bot.get_response(request)
				print('Bot: ', response)
				#bot_input = input("You: ")
				#bot_respose = bot.get_response(bot_input)
				#print(f"{BOTNAME}: {bot_respose}")

		except(KeyboardInterrupt, EOFError, SystemExit):
			break
		

if __name__ == "__main__":
	start()
	
