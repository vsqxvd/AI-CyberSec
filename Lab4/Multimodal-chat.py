import ollama
import os

MODEL = "moondream:1.8b"
#moondream:1.8b
#qwen3.5:0.8b

img_path = input("Введіть шлях до картинки: ")
if img_path.lower() == "no":
    img_path = "img.jpg"
if not os.path.exists(img_path):
    print("Файл не знайдено.")
    exit()

response = ollama.chat(
  model = MODEL,
  messages=[
    {
      'role': 'user',
      'content': "What is in this image? Be concise.",
      'images': [img_path]
    }
  ]
)
answer = response['message']['content']
print(f"Аналіз картинки: {answer}\n")

filename = "Analyse-image.txt"
with open(filename, "a", encoding="utf-8") as file:
    file.write(f"Шлях картинки: {img_path}\n")
    file.write(f"Модель: {MODEL}\n")
    file.write(f"Аналіз картинки: {answer}\n\n")
