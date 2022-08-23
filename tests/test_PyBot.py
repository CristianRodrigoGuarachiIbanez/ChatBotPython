
from chatBotPython.chatBot import LOAD_MODEL
from chatBotPython.CNN_Model import CNN
import os
path = os.path.dirname(os.path.abspath(__file__))
def start_chat():

    cnn = CNN()
    cnn.write_results(filename=path + "/chatbot.h5")
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