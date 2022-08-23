import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.stem import WordNetLemmatizer
import json
from urllib.request import urlopen
import pickle
import numpy as np
import random

class DATA:
    def __init__(self, url=None):
        # importing the GL Bot corpus file for pre-processing
        self._words = []
        self._classes = []
        self._documents = []
        self._ignore_words = ['?', '!']
        if not (url):
            url = r"https://raw.githubusercontent.com/vishal-verma27/Building-a-Simple-Chatbot-in-Python-using-NLTK/611bc5c96f25aa0e5f8e71a97c421fab7781214e/Train_Bot.json"
            intents = self._load_url_data(url)
        else:
            intents = self._load_data(url)

        self._bot_corpus(intents)
    @staticmethod
    def _load_data(path):
        data_file = open(path).read()
        return json.loads(data_file)
    @staticmethod
    def _load_url_data( url):
        response = urlopen(url)
        data_file = response.read()
        return json.loads(data_file)
    def _bot_corpus(self, intents ):
        nltk.download('punkt')
        nltk.download('wordnet')
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # tokenize each word
                w = nltk.word_tokenize(pattern)
                self._words.extend(w)
                # add documents in the corpus
                self._documents.append((w, intent['tag']))

                # add to our classes list
                if intent['tag'] not in self._classes:
                    self._classes.append(intent['tag'])


class Lemmatizer(DATA):
    def __init__(self, url=None):
        super(DATA, self).__init__(url)
        # create an object of WordNetLemmatizer
        self._lemmatizer = WordNetLemmatizer()
    def _remove_duplicated_words(self):
        words = [self._lemmatizer.lemmatize(w.lower()) for w in self._words if w not in self._ignore_words]
        return sorted(list(set(words)))
    def _get_classes(self):
        return sorted(list(set(self._classes)))
    def write(self):
        pickle.dump(self._words, open('words.pkl', 'wb'))
        pickle.dump(self._classes, open('classes.pkl', 'wb'))

class TrainingData(Lemmatizer):

    def __init__(self, url=None):
        super(TrainingData, self).__init__(url)

        # create an empty array for our output
        self._output_empty = [0] * len(self._classes)
        self._trainingSet()

    def _trainingSet(self):

        training = []

        # training set, bag of words for each sentence
        for doc in self._documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]

            # lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [self._lemmatizer.lemmatize(word.lower()) for word in pattern_words]

            # create our bag of words array with 1, if word match found in current pattern
            for w in self._words:
                bag.append(1) if w in pattern_words else bag.append(0)
            # output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(self._output_empty)
            output_row[self._classes.index(doc[1])] = 1
            training.append([bag, output_row])
        self._data_array(training)

    def _data_array(self, training):

        # shuffle features and converting it into numpy arrays
        random.shuffle(training)
        training = np.array(training)
        # create train and test lists
        self._train_x = list(training[:, 0])
        self._train_y = list(training[:, 1])

    def get_training_data(self):
        return (self._train_x, self._train_y)


