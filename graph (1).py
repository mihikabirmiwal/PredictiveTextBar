class Node(object):
    def __init__(self, id, dictionary_o={}):
        self.id = id
        self.dictionary_o = dictionary_o

    @property
    def dictionary(self):
        return self.dictionary_o

    @dictionary.setter
    def dictionary(self, dic):
        self.dictionary_o = dic
