import pickle
import numpy as np

class homebrew:
    @staticmethod
    def load_from_file(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data