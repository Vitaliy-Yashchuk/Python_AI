import ollama

def chat_with_llm_streaming():
    print("Ласкаво просимо до локального чат-бота!")
    print("Введіть ваше повідомлення або 'Bye' для виходу.\n")
    
    chat_history = []
    
    while True:
        user_input = input("Ви: ")
        
        if user_input.lower() == 'bye':
            print("Бот: До побачення! Було приємно поспілкуватися.")
            break
        
        chat_history.append({'role': 'user', 'content': user_input})
        
        try:
            print("Бот: ", end="", flush=True)
            
            full_response = ""
            stream = ollama.chat(
                model='mistral',
                messages=chat_history,
                stream=True
            )
            
            for chunk in stream:
                chunk_content = chunk['message']['content']
                print(chunk_content, end="", flush=True)
                full_response += chunk_content
            
            print()
            chat_history.append({'role': 'assistant', 'content': full_response})
            
        except Exception as e:
            print(f"\nСталася помилка: {e}")
            break

if __name__ == "__main__":
    chat_with_llm_streaming()