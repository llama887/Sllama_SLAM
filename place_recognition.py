from typing import Any, List

import cv2
import faiss
import numpy as np
from typing import List, Tuple
import os

class Image_Embedding():
    count = 0
    def __init__(self, pose : Tuple[int, int, int], image):
        self.x = pose[0]
        self.y = pose[1]
        self.heading = pose[2]
        self.image = image

class Target_Locator():
    def __init__(self):
        self.embeddings : List[Image_Embedding]= []
        self.histograms = None
        self.visual_dictionary = None
        self.save_dir = "/data/images/"
    def generate_vocabulary(self):
        print(f"Generating vocabulary")
        if self.embeddings == None:
            print(f"We're in test mode...")
            return
        if len(self.embeddings) != 0:
            print(f"Extracting features from images")
            print(
                f"\nFinding descriptors for {len(self.embeddings)} images, with {len(self.embeddings)} possible poses"
            )
            keypoints, descriptors = self._extract_sift_features()
            print(f"Creating dictionary for images")
            self.visual_dictionary = self._create_visual_dictionary(
                np.vstack(descriptors), num_clusters=100
            )
            print(f"Creating {len(self.embeddings)} histograms")
            self.histograms = self._generate_feature_histograms(descriptors)
            print(f"Done!")
        else:
            print(f"Creating embeddings from files")
            self._create_embeddings_from_files()
            print(f"Done generating embeddings")
            self.generate_vocabulary()

    
    def add_image(self, embedding : Image_Embedding):
        self.embeddings.append(embedding)
        posex, posey, poser = embedding.x, embedding.y, embedding.heading
        save_str = str(posex) + "_" + str(posey) + "_" + str(poser) + "_.jpg"
        save_dir_full = os.pardir + self.save_dir
        save_path = save_dir_full + save_str
        if not os.path.isdir(save_dir_full):
            os.makedirs(save_dir_full)
        cv2.imwrite(save_path, embedding.image)

    def _create_embeddings_from_files(self):
        save_dir_full = os.pardir + self.save_dir
        if not os.path.isdir(save_dir_full):
            self.embeddings = None
            return
        for filename in os.listdir(save_dir_full):
            save_str = save_dir_full + filename
            em_img = cv2.imread(save_str)
            pose = filename.split("_")
            posex, posey, posez = pose[0], pose[1], pose[2]
            em_pose = (posex, posey, posez)
            self.embeddings.append(Image_Embedding(em_pose, em_img))

    
    def _extract_sift_features(self):
        images = [embedding.image for embedding in self.embeddings]
        sift = cv2.SIFT_create()
        keypoints_list = []
        descriptors_list = []

        for image in images:
            keypoints, descriptors = sift.detectAndCompute(image, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)

        return keypoints_list, descriptors_list


    def _create_visual_dictionary(self, descriptors: np.ndarray, num_clusters: int) -> Any:
        d = descriptors.shape[1]  # Dimension of each vector
        kmeans = faiss.Kmeans(d=d, k=num_clusters, niter=300)
        kmeans.train(descriptors.astype(np.float32))
        return kmeans


    def _generate_feature_histograms(self, 
        descriptors: np.ndarray
    ) -> List[Any]:
        num_clusters = self.visual_dictionary.k
        histograms = []

        for desc in descriptors:
            histogram = np.zeros(num_clusters)
            _, labels = self.visual_dictionary.index.search(desc.astype(np.float32), 1)
            for label in labels.flatten():
                histogram[label] += 1
            histograms.append(histogram)
        return histograms


    def find_target_indices(self, new_image: np.ndarray) -> np.ndarray:
        # Step 1: Extract features from the new image
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(new_image, None)

        # Ensure descriptors are in the correct format (np.float32)
        descriptors = descriptors.astype(np.float32)

        # Step 2: Generate the feature histogram for the new image
        num_clusters = self.visual_dictionary.k
        histogram = np.zeros(num_clusters)

        # Use FAISS to find nearest clusters
        _, labels = self.visual_dictionary.index.search(descriptors, 1)
        for label in labels.flatten():
            histogram[label] += 1

        # Step 3: Compare the histogram to the list of histograms
        distances = [np.linalg.norm(histogram - hist) for hist in self.histograms]

        # Find the indices of the 3 best candidates
        best_candidates_indices = np.argsort(distances)[:3]

        return np.array(best_candidates_indices)
