import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from classifiers import MultiLayerPerceptron
import os
import cv2

data_file = r'datasets\faces_training.pkl'
output_preprocessed = r'datasets\faces_preprocessed.pkl'
output_mlp = r'params\mlp.xml'

with open(data_file, 'rb') as f:
    samples = pickle.load(f)
    labels = pickle.load(f)

X = np.array(samples)
y = np.array(labels)
print(f"Number of sample: {X.shape[0]}, Vector size: {X.shape[1]}, Label: {np.unique(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Number of train sample: {X_train.shape[0]}, Number of test sample: {X_test.shape[0]}")

n_components = 200 
pca = PCA(n_components=n_components, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

V = pca.components_
m = pca.mean_
print(f"Size after PCA - Train: {X_train_pca.shape}, Test: {X_test_pca.shape}")

n_classes = len(np.unique(y))
layer_sizes = np.array([n_components, 100, 50, n_classes]) 
MLP = MultiLayerPerceptron(layer_sizes, np.unique(y))
MLP.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 1000, 0.0001))
MLP.model.setBackpropMomentumScale(0.1)
MLP.model.setBackpropWeightScale(0.001) 
MLP.train(X_train_pca, y_train)

y_pred = MLP.predict(X_test_pca)
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

os.makedirs(os.path.dirname(output_preprocessed), exist_ok=True)
os.makedirs(os.path.dirname(output_mlp), exist_ok=True)
with open(output_preprocessed, 'wb') as f:
    pickle.dump(((X_train_pca, y_train), (X_test_pca, y_test), V, m), f)
MLP.save(output_mlp)
print("Saved")