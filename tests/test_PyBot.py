
from chatBotPython.chatBot import LOAD_MODEL

def start_chat():
    response = LOAD_MODEL()
    print("Bot: This is Sophie! Your Personal Assistant.\n\n")
    while True:
        inp = str(input()).lower()
        if inp.lower()=="end":
            break
        if inp.lower()== '' or inp.lower()== '*':
            print('Please re-phrase your query!')
            print("-"*50)
        else:
            print(f"Bot: {response.chatbot_response(inp)}"+'\n')
            print("-"*50)


if (__name__ == "__main__"):

    start_chat()