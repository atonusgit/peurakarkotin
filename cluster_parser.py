debug = 0

if debug >= 2: print("import os")
import os
if debug >= 2: print("import cv2")
import cv2
if debug >= 2: print("import numpy as np")
import numpy as np
if debug >= 2: print("import pandas as pd")
import pandas as pd
if debug >= 2: print("import multiprocessing as mp")
import multiprocessing as mp
if debug >= 2: print("from dotenv import load_dotenv")
from dotenv import load_dotenv
if debug >= 2: print("import matplotlib.pyplot as plt")
import matplotlib.pyplot as plt
if debug >= 2: print("from sklearn.cluster import KMeans")
from sklearn.cluster import KMeans
if debug >= 2: print("from sklearn.decomposition import PCA")
from sklearn.decomposition import PCA
if debug >= 2: print("from datetime import datetime, timedelta")
from datetime import datetime, timedelta
if debug >= 2: print("from sklearn.metrics import silhouette_score")
from sklearn.metrics import silhouette_score

load_dotenv()

def count_files_by_needle(folder, needle):
    if debug >= 2: print("count_files_by_needle()")

    i = 0
    for filename in os.listdir(folder):
        if needle in filename:
            i += 1
    return i

def load_images_from_folder(folder, needle, end_index, max_images_in_batch):
    if debug >= 2: print(f"load_images_from_folder()")

    i = 0
    filenames = []

    for filename in os.listdir(folder):
        if needle in filename:
            i += 1
            if i < end_index and i >= end_index - max_images_in_batch:
                filenames.append(filename)

    return filenames

def extract_features(img):
    if img is None:
        raise ValueError("Image is None")

    resized_img = cv2.resize(img, (32, 32))  # Resize for consistency
    # gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(_winSize=(32, 32),
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8),
                            _nbins=9)
    # h = hog.compute(gray_img)
    h = hog.compute(resized_img)
    if h is None or h.size == 0:
        raise ValueError("HOG descriptor computation failed")

    return h.flatten()

def calculate_single_cluster_radius(features):
    centroid = np.mean(features, axis=0)
    distances = np.linalg.norm(features - centroid, axis=1)
    radius = np.max(distances)
    return radius

def calculate_single_cluster_variance(features):
    centroid = np.mean(features, axis=0)
    distances = np.linalg.norm(features - centroid, axis=1)
    variance = np.mean(distances**2)
    return variance

def detect_single_cluster(features):
    if debug >= 0: print("\nDetect single cluster")

    probability = 0

    pca_radius = calculate_single_cluster_radius(features)
    if debug >= 0: print(f'pca_radius: {pca_radius}')
    if pca_radius < 1:
        probability += 1

    variances = calculate_single_cluster_variance(features)
    if debug >= 0: print(f'single_cluster_variances: {variances}')
    if variances < 0.5:
        probability += 1

    kmeans_single = KMeans(n_clusters=2, n_init=10)
    kmeans_single.fit(features)
    labels = kmeans_single.fit_predict(features)

    two_cluster_silhouette_score = silhouette_score(features, labels)
    if debug >= 0: print(f"two_cluster_silhouette_score: {two_cluster_silhouette_score}")
    if two_cluster_silhouette_score < 0.1:
        probability += 1

    two_cluster_inertia = kmeans_single.inertia_
    if debug >= 0: print(f"two_cluster_inertia: {two_cluster_inertia}")
    if two_cluster_inertia < 5:
        probability += 1

    if probability > 2:
        return True, two_cluster_silhouette_score

    return False, two_cluster_silhouette_score

def get_optimal_clusters(features, max_clusters):
    if debug >= 0: print("\nGet optimal clusters")

    is_single_cluster, two_cluster_silhouette_score = detect_single_cluster(features)

    if is_single_cluster:
        if debug >= 0: print(f"single cluster detected")
        return 1

    if two_cluster_silhouette_score < 0.25 and len(features) < 30:
        if debug >= 0: print(f"too sparce, send all images")
        return 100

    if debug >= 1: print("detect number of clusters")
    inertia = []
    silhouette_scores = []
    max_clusters = min(max_clusters, len(features))
    for k in range(2, max_clusters+1):
        if debug >= 1: print(f"calculate for {k} clusters")
        kmeans = KMeans(n_clusters=k, n_init=10)
        if debug >= 1: print("run kmeans.fit()")
        kmeans.fit(features)
        if debug >= 1: print("append kmeans.inertia_")
        inertia.append(kmeans.inertia_)
        if debug >= 1: print("run silhouette_score()")
        score = silhouette_score(features, kmeans.labels_)
        if debug >= 1: print(f"silhouette score: {score}")
        silhouette_scores.append(score)

    if debug >= 1: print("get optimal_cluster_amount")
    optimal_cluster_amount = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2

    if debug >= 0: print(f'optimal number of clusters: {optimal_cluster_amount}')
    return optimal_cluster_amount

def get_features(images):
    if debug >= 0: print("\nGet features")

    # Process images in batches to save memory
    batch_size = 50
    features = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]

        with mp.Pool(2) as pool:  # Limit to 2 processes to save memory
            batch_features = pool.map(extract_features, batch_images)

        features.extend(batch_features)

    features = np.array(features)

    return features

def reduce_features_with_pca(features):
    if debug >= 0: print("\nReduce features with PCA - Principal Component Analysis")

    n_components = min(20, len(features))
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)

    return pd.DataFrame(reduced_features)

def parse_centroid_images(features, filenames, labels, centroids, optimal_cluster_amount):
    if debug >= 0: print("\nGet centroid images")

    centroid_images = {}
    for cluster_id in range(optimal_cluster_amount):
        centroid = centroids[cluster_id]
        min_distance = float('inf')
        centroid_image_index = -1
        for idx, feature in enumerate(features):
            if labels[idx] == cluster_id:
                distance = np.linalg.norm(feature - centroid)
                if distance < min_distance:
                    min_distance = distance
                    centroid_image_index = idx
        centroid_images[cluster_id] = filenames[centroid_image_index]

    return centroid_images

def cluster_and_get_labels_and_centroids(optimal_cluster_amount, features):
    if debug >= 0: print("\nCluster and get labels and centroids")

    kmeans = KMeans(n_clusters=optimal_cluster_amount, n_init=10)
    labels = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_

    return labels, centroids

def cluster_and_get_centroid_images(images, filenames, min_images_for_clustering):
    if debug >= 0: print("\nCluster and get centroid images")

    features = get_features(images)
    reduced_features = reduce_features_with_pca(features)
    optimal_cluster_amount = get_optimal_clusters(reduced_features, min_images_for_clustering)  # Reduce the max number of clusters to save time

    if optimal_cluster_amount == 100:
        return {i: filenames[i] for i in range(len(filenames))}

    labels, centroids = cluster_and_get_labels_and_centroids(optimal_cluster_amount, reduced_features)
    centroid_images = parse_centroid_images(reduced_features, filenames, labels, centroids, optimal_cluster_amount)

    return centroid_images

def get_centroid_filenames(filenames, date_str, min_images_for_clustering, cropped_plot_images_folder_path):
    if debug >= 0: print("\nGet centroid filenames")

    do_clustering = True

    if not filenames:
        if debug >= 0: print(f"no images found for {date_str}")
        exit(0)

    if len(filenames) < min_images_for_clustering:
        if debug >= 0: print(f"no clustering for {len(filenames)} images")
        centroid_filenames = filenames
        do_clustering = False

    if do_clustering:
        if debug >= 1: print("cv2.imread() images")
        images = [cv2.imread(os.path.join(cropped_plot_images_folder_path, fname)) for fname in filenames]

        if debug >= 1: print("cluster_and_get_centroid_images()")
        centroid_images = cluster_and_get_centroid_images(images, filenames, min_images_for_clustering)
        centroid_filenames = []

        for label, filename in centroid_images.items():
            centroid_filenames.append(filename)

    return centroid_filenames

def send_to_dropbox(first_iteration_images_path, destination_images_path):
    if debug >= 0: print("\nSend to dropbox")

    os.system(f'cd {os.getenv("ROOT_DIRECTORY")} && bash tools/send_to_dropbox.sh havainnot-sensitive/cluster_centroids && rm -rf {first_iteration_images_path}/* && rm -rf {destination_images_path}/*')

if __name__ == '__main__':
    if debug >= 0: print("Starting script...")

    cropped_plot_images_folder_path = os.getenv("CROPPED_PLOT_IMAGES_FOLDER_PATH")
    large_images_path = os.getenv("LARGE_IMAGES_PATH")
    first_iteration_images_path = os.getenv("FIRST_ITERATION_IMAGES_PATH")
    destination_images_path = os.getenv("DESTINATION_IMAGES_PATH")

    max_images_in_batch = 500
    min_images_for_clustering = 4

    if debug >= 1: print("\nSet target_date")
    target_date = datetime.now() - timedelta(1) # yesterday
    date_str = target_date.strftime('%y%m%d')
    if debug >= 0: print(f"target_date set to {date_str}")

    if debug >= 0: print("\n")
    if debug >= 0: print("****************************************")
    if debug >= 0: print("*           First iteration            *")
    if debug >= 0: print("****************************************")

    file_count = count_files_by_needle(cropped_plot_images_folder_path, date_str)
    if debug >= 0: print(f"file_count: {file_count} (for 1st iteration images)")

    for i in range(0, file_count, max_images_in_batch):
        if debug >= 2: print(f"\nload_images_from_folder(), iteration {i}")
        filenames = load_images_from_folder(cropped_plot_images_folder_path, date_str, i + max_images_in_batch, max_images_in_batch)
        if debug >= 2: print(f"get_centroid_filenames(), iteration {i}")
        centroid_filenames = get_centroid_filenames(filenames, date_str, min_images_for_clustering, cropped_plot_images_folder_path)

        if debug >= 1: print("\ncopy centroid images to first_iteration_images_path")
        for filename in centroid_filenames:
            if debug >= 0: print(f"copying: {filename}")
            os.system(f'cp {cropped_plot_images_folder_path}/{filename} {first_iteration_images_path}/')

    if debug >= 0: print("\n")
    if debug >= 0: print("****************************************")
    if debug >= 0: print("*           Second iteration           *")
    if debug >= 0: print("****************************************")

    file_count = count_files_by_needle(first_iteration_images_path, date_str)
    if debug >= 0: print(f"file_count: {file_count} (for 2nd iteration images)")

    filenames = load_images_from_folder(first_iteration_images_path, date_str, max_images_in_batch, max_images_in_batch)
    centroid_filenames = get_centroid_filenames(filenames, date_str, min_images_for_clustering, first_iteration_images_path) #cropped_plot_images_folder_path)

    if debug >= 1: print("\ncopy centroid images to destination_images_path")
    for filename in centroid_filenames:
        if debug >= 0: print(f"copying: {filename}")

        cropped_plot_index = filename.find('_cropped_plot.jpg')
        if cropped_plot_index != -1:
            filename_main = filename.replace('_cropped_plot.jpg', '.jpg')
            filename_plotted = filename.replace('_cropped_plot.jpg', '_plotted.jpg')
            filename_for_copying = filename.replace('_cropped_plot.jpg', '*')

        cropped_plot_index = filename.find('_cropped_plot2.jpg')
        if cropped_plot_index != -1:
            filename_main = filename.replace('_cropped_plot2.jpg', '.jpg')
            filename_plotted = filename.replace('_cropped_plot2.jpg', '_plotted.jpg')
            filename_for_copying = filename.replace('_cropped_plot2.jpg', '*')

        os.system(f'cp {cropped_plot_images_folder_path}/{filename} {destination_images_path}/')
        os.system(f'cp {large_images_path}/{filename_for_copying} {destination_images_path}/')

    send_to_dropbox(first_iteration_images_path, destination_images_path)
