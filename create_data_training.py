import os
import cv2
import pickle
import numpy as np

dataset_dir = r'data\train'  
output_file = r'datasets\faces_training.pkl'  

emotions = ['neutral', 'happy', 'sad', 'surprise', 'angry', 'disgusted']
label_map = {emotion: idx for idx, emotion in enumerate(emotions)}

folder_to_emotion = {
    'neutral': 'neutral',
    'happy': 'happy',
    'sad': 'sad',
    'surprise': 'surprise',
    'angry': 'angry',
    'disgusted': 'disgusted'
}

samples = []
labels = []

for folder_name in os.listdir(dataset_dir):
    if folder_name not in folder_to_emotion:
        #print(f"Folder {folder_name} did not match any emotion, skip")
        continue
    
    emotion = folder_to_emotion[folder_name]
    emotion_dir = os.path.join(dataset_dir, folder_name)
    if not os.path.isdir(emotion_dir):
        #print(f"{emotion_dir} is not a folder, skip")
        continue
    
    for img_file in os.listdir(emotion_dir):
        img_path = os.path.join(emotion_dir, img_file)
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            #print(f"Can not read {img_path}, skip")
            continue
        
        print(f"Image size {img_path}: {img.shape}")
        
        resized_img = cv2.resize(img, (100, 100))
        processed_frame = resized_img.flatten().astype(np.float32)
        
        samples.append(processed_frame)
        labels.append(label_map[emotion])
        #print(f"Processed {img_path} - Emotion: {emotion}")

if samples:
    os.makedirs(os.path.dirname(output_file), exist_ok=True) 
    with open(output_file, 'wb') as f:
        pickle.dump(samples, f)
        pickle.dump(labels, f)
    print(f"Saved {len(samples)} samples in {output_file}")
else:
    print("Saved nothing")