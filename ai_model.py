import spacy
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Create ChatBot Instance
chatbot = ChatBot('AgriBot')

# Train with Corpus Data
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")

def get_response(user_input):
    response = chatbot.get_response(user_input)
    return str(response)
