import matplotlib.pyplot as plt
import numpy as np
import random

ITERATIONS = 1000
INIT_POINTS = 20

def offset_calculation(coordinate):
    offset = 100
    if 5000 - abs(coordinate) < 100:
        offset = (5000 - abs(coordinate)) // 2
    return offset

def gen_offset(x, y):
    x_offset = offset_calculation(x)
    y_offset = offset_calculation(y)
    x = x + random.randint(-x_offset, x_offset)
    y = y + random.randint(-y_offset, y_offset)
    return np.array([x, y])  # Return as a NumPy array

def init_points():
    coordinates_set = set()
    coordinates_array = []

    while len(coordinates_set) < INIT_POINTS:
        x = np.random.randint(-5000, 5000)
        y = np.random.randint(-5000, 5000)
        point = np.array([x, y])
        coordinates_set.add(tuple(point))
        coordinates_array.append(point)

    while len(coordinates_array) < ITERATIONS + INIT_POINTS:
        random_coordinate = random.choice(coordinates_array)
        x, y = random_coordinate
        new_point = gen_offset(x, y)
        if tuple(new_point) not in coordinates_set:
            coordinates_set.add(tuple(new_point))
            coordinates_array.append(new_point)

    return np.array(coordinates_array)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def distance_matrix(centroids):
    # num_points = centroids.shape[0]
    num_points = len(centroids)
    dist_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = euclidean_distance(centroids[i], centroids[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    np.fill_diagonal(dist_matrix, np.inf)
    return dist_matrix

def agglomerative_centroid(clusters, target_clusters=10):
    while len(clusters) > target_clusters:
        centroids = []
        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            centroids.append(centroid)

        dist_matrix = distance_matrix(np.array(centroids))

        cluster1, cluster2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        clusters[cluster1].extend(clusters[cluster2])
        del clusters[cluster2]
        print(f"Merged clusters {cluster1} and {cluster2}, Total clusters remaining: {len(clusters)}")

    final_centroids = []
    for cluster in clusters:
        final_centroid = np.mean(cluster, axis=0)
        final_centroids.append(final_centroid)

    print("Final centroids:", final_centroids)
    return np.array(final_centroids), clusters


def show_clusters(final_centroids, clusters):
    plt.figure(figsize=(8, 8))
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)

    colors = plt.colormaps.get_cmap('tab10', len(clusters))

    for idx, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        centroid = np.mean(cluster_points, axis=0)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=5, color=colors(idx), label=f"Cluster {idx + 1}")
        plt.scatter(centroid[0], centroid[1], s=100, color=colors(idx), edgecolor='black', marker='X')

    plt.title("Agglomerative Clustering Visualization with Centroids")
    plt.legend()
    plt.show()


def main():
    random.seed(50)
    np.random.seed(50)
    coordinates = init_points()
    clusters = [[point] for point in coordinates]
    final_centroids, final_clusters = agglomerative_centroid(clusters, target_clusters=5)
    show_clusters(final_centroids, final_clusters)


if __name__ == "__main__":
    main()