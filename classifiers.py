import cv2
import numpy as np

class MultiLayerPerceptron:
    def __init__(self, layer_sizes, labels):
        self.labels = labels
        self.model = cv2.ml.ANN_MLP_create()
        self.model.setLayerSizes(layer_sizes)
        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
        self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001))

    def train(self, samples, labels):
        labels_one_hot = np.zeros((labels.shape[0], len(self.labels)), dtype=np.float32)
        for i, label in enumerate(labels):
            labels_one_hot[i, label] = 1.0

        self.model.train(samples, cv2.ml.ROW_SAMPLE, labels_one_hot)

    def predict(self, samples):
        _, results = self.model.predict(samples)
        predictions = np.argmax(results, axis=1)
        return predictions

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = cv2.ml.ANN_MLP_load(filename)