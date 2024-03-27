import redis
import os
from typing import Any, List

import cv2
import faiss
import numpy as np
import random

class Image_Embedding():
    count = 0
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image
        self.count = Image_Embedding.count
        Image_Embedding.count += 1

class Vocabulary_Generator():
    def __init__(self):
        self.images = []
        self.embeddings : List[Image_Embedding]= []
    def generate_vocabulary(self):
        if len(self.images) != 0:
            if self.self_validate:
                index = random.randint(10, len(self.images) - 10)
                self.validation_img = self.images[index]
                for i in range(-5, 5):
                    del self.images[index + i]
                    del self.poses[index + i]

            print(
                f"\nFinding descriptors for {len(self.images)} images, with {len(self.poses)} possible poses"
            )
            keypoints, descriptors = self._extract_sift_features(self.images)
            print(f"Creating dictionary for images")
            self.visual_dictionary = self._create_visual_dictionary(
                np.vstack(descriptors), num_clusters=100
            )
            print(f"Creating {len(self.images)} histograms")
            self.histograms = self._generate_feature_histograms(
                descriptors, self.visual_dictionary
            )
    
    def add_image(self, image, pose):
        self.images.append(image)
        self.poses.append(pose)

    def _load_images_from_folder(self) -> List[Any]:
        images = []
        for filename in os.listdir(self.path):
            img = cv2.imread(os.path.join(self.path, filename))

            if img is not None:
                images.append(img)
        return images
    
    def _extract_sift_features(images: List):
        sift = cv2.SIFT_create()
        keypoints_list = []
        descriptors_list = []

        for image in images:
            keypoints, descriptors = sift.detectAndCompute(image, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)

        return keypoints_list, descriptors_list


    def _create_visual_dictionary(descriptors: np.ndarray, num_clusters: int) -> Any:
        d = descriptors.shape[1]  # Dimension of each vector
        kmeans = faiss.Kmeans(d=d, k=num_clusters, niter=300)
        kmeans.train(descriptors.astype(np.float32))
        return kmeans


    def _generate_feature_histograms(
        descriptors: np.ndarray, visual_dictionary: Any
    ) -> List[Any]:
        num_clusters = visual_dictionary.k
        histograms = []

        for desc in descriptors:
            histogram = np.zeros(num_clusters)
            _, labels = visual_dictionary.index.search(desc.astype(np.float32), 1)
            for label in labels.flatten():
                histogram[label] += 1
            histograms.append(histogram)

        return histograms


    def _process_image_and_find_best_match(
        new_image: np.ndarray, list_of_histograms: List[Any], kmeans: Any
    ) -> np.ndarray:
        # Step 1: Extract features from the new image
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(new_image, None)

        # Ensure descriptors are in the correct format (np.float32)
        descriptors = descriptors.astype(np.float32)

        # Step 2: Generate the feature histogram for the new image
        num_clusters = kmeans.k
        histogram = np.zeros(num_clusters)

        # Use FAISS to find nearest clusters
        _, labels = kmeans.index.search(descriptors, 1)
        for label in labels.flatten():
            histogram[label] += 1

        # Step 3: Compare the histogram to the list of histograms
        distances = [np.linalg.norm(histogram - hist) for hist in list_of_histograms]

        # Find the indices of the 3 best candidates
        best_candidates_indices = np.argsort(distances)[:3]

        return np.array(best_candidates_indices)
