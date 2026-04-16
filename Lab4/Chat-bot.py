import ollama

MODEL = "moondream:1.8b"
#moondream:1.8b
#qwen3.5:0.8b не працює коректно з чатом ні через стрімінг, ні через повну відповідь

DEMENTIA = 10
memory = []

def chat():
    print("Чат-бот запущено. Для виходу напиши 'exit'.\n")
    print("Вітаю! Що хочете спитати?")

    while True:
        user_input = input("Питання: ")
        if user_input.lower() == "exit":
            print("Завершення роботи чату.")
            break

        memory.append({'role':'user', 'content': user_input})
        if len(memory) > DEMENTIA:
            memory.pop(0)

        response = ollama.chat(
            model = MODEL,
            messages = memory,
            stream = True
        )

        print(f"Відповідь: ", end = "")

        save_memory = ""

        for block in response:
            content = block['message']['content']
            print(content, end = "", flush = True)
            save_memory += content
        print()

        memory.append({'role':'assistant', 'content': save_memory})
        save_to_file(user_input, save_memory)

def save_to_file(user_input, answer):
    filename = "Chat-with-bot.txt"

    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"Model use: {MODEL}\n")
        file.write(f"User: {user_input}\n")
        file.write(f"Bot: {answer}\n\n")

if __name__ == "__main__":
    chat()
