import pickle
import urllib.request
from enum import Enum
from graph import Node
from random import random
from final_tests import windowed

""" Builds a statistical model of some data and then outputs a stream of
data that is similar to the original but randomly generated.

Formally, building a Markov Chain, a directed graph where every node 
is a state and every outgoing edge from a node is a possible token to
find while in that state. The edges have probabilities that define how
likely it is to follow that edge. The probability of all the edges leaving 
a node sum to 1. """


class Tokenization(Enum):
    word = 1
    character = 2
    byte = 3
    none = 4

# A Markov chain based random data generator that can save its model.
class RandomWriter(object):

    # Initialize a random writer.
    def __init__(self, level=0, tokenization=None):
        self.level = level
        self.tokenization = tokenization
        self.nodes = {}  # statistical model

    def save_pickle(self, filename_or_file_object):
        fi = self.get_file_object(filename_or_file_object, "wb")
        pickle.dump(self.nodes, fi)
        fi.close()

    @classmethod
    def load_pickle(cls, filename_or_file_object):
        fi = cls.get_file_object(filename_or_file_object, "rb")
        rw = cls()
        rw.nodes = pickle.load(fi)
        return rw

    @staticmethod
    def get_file_object(filename_or_file_object, op):
        if isinstance(filename_or_file_object, str):
            fi = open(filename_or_file_object, op)
        else:
            fi = filename_or_file_object
        return fi

    # Compute the probabilities based on the data downloaded from url.
    def train_url(self, url):
        if self.tokenization in [Tokenization.word, Tokenization.character, Tokenization.byte]:
            if self.tokenization in [Tokenization.word, Tokenization.character]:  # must convert data to a string
                with urllib.request.urlopen(url) as inp:
                    data = inp.read()
                    data = data.decode("utf-8")
            else:  # data must be bytes
                data = urllib.request.urlopen(url)
                data = data.read()
            self.train_iterable(data)

    # Compute the probabilities based on the data given.
    def train_iterable(self, data):
        if self.tokenization in [Tokenization.word, Tokenization.character] and not isinstance(data, str):
            raise TypeError
        if self.tokenization == Tokenization.byte and not isinstance(data, bytes):
            raise TypeError
        if self.tokenization == Tokenization.none and not hasattr(data, '__iter__'):
            raise TypeError

        # makes the data an iterable, which is passed into train_helper
        data_m = data
        if self.tokenization == Tokenization.word:
            data_m = data.split(" ")
        if self.tokenization == Tokenization.character:
            data_m = list(data)
        if self.tokenization == Tokenization.byte:
            data_m = []
            for x in range(len(data)):
                data_m.append(data[x])
        self.train_helper(data_m)

    # data parameter will be an iterable
    def train_helper(self, data):
        for window in windowed(data, self.level + 1):
            initial_node = window[:self.level]
            next_node = window[-self.level:]
            if initial_node not in self.nodes:
                self.nodes[initial_node] = Node(initial_node, {next_node: 1})
            else:
                node = self.nodes[initial_node]
                node_dict = node.dictionary
                if next_node not in node_dict:
                    node_dict[next_node] = 1
                else:
                    node_dict[next_node] += 1
                node.dictionary = node_dict
        if next_node not in self.nodes:
            self.nodes[next_node] = Node(next_node, {(list(self.nodes.keys()))[0]: 1})

    # returns next token, not node
    # Yield random tokens using the model
    def generate(self, last_node=None):
        while True:
            next_node = self.generate_node(last_node)
            last_node = next_node
            yield next_node[-1]

    # generates a single token based on the last Node
    def generate_node(self, last_node=None):
        if last_node is None:
            last_node = list(self.nodes.keys())[0]
        node_dict = self.nodes[last_node].dictionary
        new_dict = {}
        total_occurrences = sum(node_dict.values())
        displacement = 0
        for key, value in node_dict.items():
            new_dict[key] = (1.0 * value / total_occurrences) + displacement
            displacement = new_dict[key]
        val = random()
        for key, value in new_dict.items():
            if val < value:
                return key

    # Write a file using the model.
    def generate_file(self, filename, amount):
        if self.tokenization == Tokenization.byte:
            with open(filename, "wb") as fi:
                generated = list(self.nodes.keys())[0]
                for x in range(amount):
                    next_gen = self.generate_node(generated)
                    fi.write(bytes([(next_gen[-1])]))
                    generated = next_gen
            fi.close()
        else:
            with open(filename, "w") as fi:
                generated = list(self.nodes.keys())[0]
                for x in range(amount):
                    next_gen = self.generate_node(generated)
                    fi.write(str(next_gen[-1]))
                    if self.tokenization == Tokenization.word or self.tokenization == Tokenization.none:
                        fi.write(" ")
                    generated = next_gen
            fi.close()
