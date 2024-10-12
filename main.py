import matplotlib.pyplot as plt
import random
import numpy as np

ITERATIONS = 20000
INIT_POINTS = 20
#coordinate_list = np.zeros((0, 2))  # Start with an empty array


def create_point(x, y):
    plt.plot(x, y, 'o', markersize=1, markeredgecolor='black')


def init_points(coordinates):
    for i in range(INIT_POINTS):
        x = random.randint(-5000, 5000)
        y = random.randint(-5000, 5000)
        coordinates.append([x, y])


def offset_calculation(coordinate):
    offset = 100
    if 5000 - abs(coordinate) < 100:
        offset = (5000 - abs(coordinate)) // 2
    print(offset)
    return offset


def gen_offset(x, y):
    a = offset_calculation(x)
    b = offset_calculation(y)
    x_offset = max(-5000, min(5000, x + random.randint(-a, a)))
    y_offset = max(-5000, min(5000, y + random.randint(-b, b)))
    return [x_offset, y_offset]


def main_part(coordinates):
    for i in range(ITERATIONS):
        random_index = np.random.randint(len(coordinates))
        x, y = coordinates[random_index]
        new_point = gen_offset(x, y)
        coordinates.append(new_point)


def generate_points(coordinates):
    for coordinate in coordinates:
        create_point(coordinate[0], coordinate[1])


def main():
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)

    coordinates = []  # Use a list to store points
    init_points(coordinates)
    main_part(coordinates)

    generate_points(coordinates)
    plt.show()


# Entry point
if __name__ == "__main__":
    main()
