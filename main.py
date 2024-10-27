import matplotlib.pyplot as plt
import numpy as np
import math
import random
import heapq

ITERATIONS = 10000
INIT_POINTS = 20

def offset_calculation(coordinate):
    offset = 100
    if 5000 - abs(coordinate) < 100:
        offset = (5000 - abs(coordinate)) // 2
    return offset


def gen_offset(x, y):
    x_offset = offset_calculation(x)
    y_offset = offset_calculation(y)
    x = x + np.random.randint(-x_offset, x_offset)
    y = y + np.random.randint(-y_offset, y_offset)
    return np.array([x, y])  # Return as a NumPy array

def init_points():
    coordinates_set = set()  # Use a set to store unique points
    coordinates_array = []  # Use a list to store points as NumPy arrays

    # Initialize with random unique points
    while len(coordinates_set) < INIT_POINTS:
        x = np.random.randint(-5000, 5000)
        y = np.random.randint(-5000, 5000)
        point = np.array([x, y])  # Create a NumPy array
        coordinates_set.add(tuple(point))  # Store as tuple for uniqueness check
        coordinates_array.append(point)  # Store as NumPy array

    # Generate new points until the desired total count is reached
    while len(coordinates_array) < ITERATIONS + INIT_POINTS:
        random_coordinate = random.choice(coordinates_array)
        x, y = random_coordinate
        new_point = gen_offset(x, y)

        # Check for uniqueness using the set
        if tuple(new_point) not in coordinates_set:
            coordinates_set.add(tuple(new_point))  # Add as a tuple for the set
            coordinates_array.append(new_point)  # Keep the new point as a NumPy array

    return np.array(coordinates_array)  # Convert the list of arrays to a NumPy array

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def compute_centroid(cluster):
    return np.mean(cluster, axis=0)

def evaluate_clusters(clusters, threshold=500):
    for cluster in clusters:
        centroid = compute_centroid(cluster)
        avg_distance = np.mean([euclidean_distance(point, centroid) for point in cluster])
        if avg_distance > threshold:
            return False  # Stop if any cluster exceeds the threshold
    return True  # Continue if all clusters meet the threshold

def distance_matrix(clusters):
    num_clusters = len(clusters)
    dist_matrix = np.full((num_clusters, num_clusters), np.inf)

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            dist_matrix[i, j] = euclidean_distance(compute_centroid(clusters[i]), compute_centroid(clusters[j]))
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix

def agglomerative_centroid(clusters):
    target_clusters = 10
    dist_matrix = distance_matrix(clusters)  # Initialize the distance matrix

    while True:
        if not evaluate_clusters(clusters):
            print("Threshold exceeded; stopping.")
            break  # Stop if any cluster exceeds the threshold
        # Find the indices of the two closest clusters
        cluster1, cluster2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        # Merge the two closest clusters
        print(cluster1, cluster2, len(clusters))
        clusters[cluster1].extend(clusters[cluster2])
        del clusters[cluster2]  # Remove the merged cluster

        dist_matrix = np.delete(dist_matrix, cluster2, axis=0)  # Remove the row for cluster2
        dist_matrix = np.delete(dist_matrix, cluster2, axis=1)  # Remove the column for cluster2

        # Update the distance matrix
        # Set the distance from the merged cluster to itself to inf
        dist_matrix[cluster1, cluster1] = np.inf

        # Update distances for the merged cluster with respect to the remaining clusters
        for i in range(len(clusters)):
            if i != cluster1:  # Avoid updating the distance to itself
                dist_matrix[cluster1, i] = euclidean_distance(compute_centroid(clusters[cluster1]),
                                                              compute_centroid(clusters[i]))
                dist_matrix[i, cluster1] = dist_matrix[cluster1, i]  # Symmetric update

        # Print the merging information
        print(f"Merged clusters {cluster1} and {cluster2}, Total clusters remaining: {len(clusters)}")

    final_centroids = [compute_centroid(cluster) for cluster in clusters]
    return np.array(final_centroids), clusters


def show_clusters(clusters):
    plt.figure(figsize=(8, 8))
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)

    colors = plt.get_cmap('tab10', len(clusters))  # Get the colormap

    for idx, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        centroid = np.mean(cluster_points, axis=0)

        # Use the colormap with the index
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=5, color=colors(idx), label=f"Cluster {idx + 1}")
        plt.scatter(centroid[0], centroid[1], s=100, color=colors(idx), edgecolor='black', marker='o')

    plt.title("Agglomerative Clustering Visualization with Centroids")
    plt.legend()
    plt.show()


def main():
    random.seed(46)
    np.random.seed(46)
    coordinates = init_points()
    # Initialize each point as a separate cluster
    clusters = [[point] for point in coordinates]
    # Run agglomerative clustering
    final_centroids, final_clusters = agglomerative_centroid(clusters)

    # Visualize the clusters and their centroids
    show_clusters(final_clusters)  # Pass the list of clusters


if __name__ == "__main__":
    main()
