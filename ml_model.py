import random
import json
import torch
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from transformers import pipeline

nltk.download('punkt')

# Load pre-trained model for NLP
nlp_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Load predefined responses
responses = {
    "best crops": "For Telangana, the best crops include rice, maize, and pulses.",
    "soil quality": "Soil testing is recommended for better yield. You should check nitrogen and pH levels.",
    "fertilizer recommendation": "For maize, nitrogen-based fertilizers are recommended.",
    "irrigation": "For optimal irrigation, 500-700 liters per hectare is recommended.",
}

def get_response(user_input):
    user_input = user_input.lower()
    for key in responses:
        if key in user_input:
            return responses[key]
    return nlp_pipeline(user_input, max_length=50, num_return_sequences=1)[0]['generated_text']
