from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login
from .forms import SignupForm
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ai_model import get_response
from django.http import JsonResponse
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import json


def home(request):
    return render(request, 'home.html')

def login_view(request):
    return render(request, 'login.html')

def signup_view(request):
    return render(request, 'signup.html')

def about_view(request):
    return render(request, 'about.html')

def contact_view(request):
    return render(request, 'contact.html')

def user_login(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("home")  # Redirect to home after login
        else:
            messages.error(request, "Invalid username or password.")
    
    return render(request, "login.html")



def signup(request):
    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data["password"])
            user.save()
            login(request, user)
            return redirect("home")  # Redirect to home after signup
    else:
        form = SignupForm()
        
    
# Initialize ChatBot
chatbot = ChatBot(
    "AgriSmartBot",
    storage_adapter="chatterbot.storage.SQLStorageAdapter",
    database_uri="sqlite:///db.sqlite3"
)

# Train the chatbot with agriculture-related data
trainer = ListTrainer(chatbot)
training_data = [
    "What are the best crops for summer?",
    "In summer, crops like maize, rice, and millet are ideal because they tolerate heat.",
    
    "How can I improve soil fertility?",
    "You can improve soil fertility by adding organic compost, rotating crops, and avoiding overuse of chemicals.",
    
    "What is the best irrigation method for wheat?",
    "Drip irrigation and sprinkler systems are effective for wheat farming.",
    
    "How can I prevent pests in my farm?",
    "Use organic pesticides, rotate crops, and introduce natural predators to control pests.",
    
    "What fertilizers are good for paddy fields?",
    "Nitrogen-based fertilizers like urea and phosphorus-rich fertilizers like DAP are recommended for paddy fields."
]
trainer.train(training_data)

@csrf_exempt
def chatbot_response(request):
    if request.method == "POST":
        try:
            # Parse the incoming request data
            data = json.loads(request.body)
            user_message = data.get("message", "").strip()
            if not user_message:
                return JsonResponse({"response": "Please enter a valid question."})

            # Get the chatbot's response
            bot_response = chatbot.get_response(user_message)
            return JsonResponse({"response": str(bot_response)})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ml_model import get_response

@csrf_exempt
def chat(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_message = data.get("message", "")
        bot_response = get_response(user_message)
        return JsonResponse({"response": bot_response})




