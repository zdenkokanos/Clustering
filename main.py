import matplotlib.pyplot as plt
import numpy as np
import random

ITERATIONS = 20000
INIT_POINTS = 20

def init_points(coordinates):
    while len(coordinates) < INIT_POINTS:
        x = np.random.randint(-5000, 5000)
        y = np.random.randint(-5000, 5000)
        coordinates.add((x, y))
    for i in range(ITERATIONS):
        random_coordinate = random.choice(list(coordinates))
        x, y = random_coordinate
        coordinates.add((gen_offset(x, y)))
    return np.array(list(coordinates))

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
    return x, y


def agglomerative_centroid(coordinates):


def show_clusters(coordinates_array):
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.scatter(coordinates_array[:, 0], coordinates_array[:, 1], s=1, edgecolor='black')
    plt.show()

def main():
    coordinates = set()  # Use a list to store points
    coordinates = init_points(coordinates)
    agglomerative_centroid(coordinates)
    show_clusters(coordinates)


# Entry point
if __name__ == "__main__":
    main()
