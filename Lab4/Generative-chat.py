import ollama

MODEL = "moondream:1.8b"
#moondream:1.8b
#qwen3.5:0.8b

user_input = input("Згенеруй: ")

response = ollama.generate(MODEL, user_input)
answer = response['response']
print(f"Відповідь: {answer}\n")

filename = "Generate-response.txt"
with open(filename, "a", encoding="utf-8") as file:
    file.write(f"Модель: {MODEL}\n")
    file.write(f"Питання: {user_input}\n\n")
    file.write(f"Відповідь: {answer}\n\n")

