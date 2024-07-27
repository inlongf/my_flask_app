import cv2
import numpy as np
import os

# Initialize a dictionary to store indexed features
index = {}

def extract_features(image):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(image, None)
    return kp, des

def index_images(folder_path):
    global index
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            kp, des = extract_features(image)
            index[filename] = des

def search_similar_features(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    kp, des = extract_features(image)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    results = {}
    for filename, indexed_des in index.items():
        matches = bf.match(des, indexed_des)
        matches = sorted(matches, key=lambda x: x.distance)
        results[filename] = len(matches)
    
    return sorted(results.items(), key=lambda item: item[1], reverse=True)
