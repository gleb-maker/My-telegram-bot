import telebot
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKEN = "Your Telegram Bot Token"
bot = telebot.TeleBot(TOKEN)
bot = telebot.TeleBot(TOKEN)

print("Download TinyLlama (~2 ГБ)...")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Bot work!")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text

    inputs = tokenizer(user_input, return_tensors="pt").to(device)

    input_ids = inputs["input_ids"]
    outputs = model.generate(**inputs, max_new_tokens=150)

    generated_ids = outputs[0][input_ids.shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    bot.send_message(message.chat.id, answer)

bot.polling(non_stop=True)
