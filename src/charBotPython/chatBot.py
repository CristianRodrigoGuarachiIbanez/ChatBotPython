import json, pickle, random
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from numpy as np

class LOAD_MODEL:
    def __init__(self):
        self._lemmatizer = WordNetLemmatizer()
        self._load_model()
    def _load_model(self, h5_file='chatbot.h5', json_file="Train_Bot.json", pkl_file='words.pkl', class_pkl='classes.pkl'):
        # load the saved model file
        self._model = load_model(h5_file)
        self._intents = json.loads(open(json_file).read())
        self._words = pickle.load(open(pkl_file, 'rb'))
        self._classes = pickle.load(open(class_pkl, 'rb'))

    def _clean_up_sentence(self, sentence):
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)

        # stem each word - create short form for word
        sentence_words = [self._lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def _bow(self, sentence, words, show_details=True):

        # tokenize the pattern
        sentence_words = self._clean_up_sentence(sentence)

        # bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:

                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return (np.array(bag))

    def _predict_class(self, sentence, model):

        # filter out predictions below a threshold
        p = self._bow(sentence, self._words, show_details=False)
        res = model.predict(np.array([p]))[0]
        error = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > error]

        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []

        for r in results:
            return_list.append({"intent": self._classes[r[0]], "probability": str(r[1])})
        return return_list

    def _getResponse(self, ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if (i['tag'] == tag):
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self, text):
        ints = self._predict_class(text, self._model)
        res = self._getResponse(ints, self._intents)
        return res

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
