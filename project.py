import random
import math
import matplotlib.pyplot as plt
import pandas as pd


# Common interface for sorting algorithms
class SortAlgorithm:
    def sort(self, data):

        raise NotImplementedError


class InsertionSort(SortAlgorithm):
    def sort(self, data):
        steps = 0
        for i in range(1, len(data)):
            key = data[i]
            j = i - 1
            while j >= 0:
                steps += 1  # comparison
                if data[j] > key:
                    data[j + 1] = data[j]
                    steps += 1  # swap/assignment
                    j -= 1
                else:
                    break
            data[j + 1] = key
            steps += 1  # final assignment
        return steps


class MergeSort(SortAlgorithm):
    def sort(self, data):
        self.steps = 0
        self.merge_sort(data, 0, len(data) - 1)
        return self.steps

    def merge_sort(self, arr, l, r):
        if l < r:
            m = (l + r) // 2
            self.merge_sort(arr, l, m)
            self.merge_sort(arr, m + 1, r)
            self.merge(arr, l, m, r)

    def merge(self, arr, l, m, r):
        L = arr[l:m + 1]
        R = arr[m + 1:r + 1]
        i, j, k = 0, 0, l
        while i < len(L) and j < len(R):
            self.steps += 1  # comparison
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
                self.steps += 1  # assignment
            else:
                arr[k] = R[j]
                j += 1
                self.steps += 1  # assignment
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
            self.steps += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
            self.steps += 1


class BubbleSort(SortAlgorithm):
    def sort(self, data):
        steps = 0
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                steps += 1  # comparison
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
                    steps += 3  # two assignments for swap (a temp variable version would be 3 steps)
        return steps


class QuickSort(SortAlgorithm):
    def sort(self, data):
        self.steps = 0
        self.quick_sort(data, 0, len(data) - 1)
        return self.steps

    def quick_sort(self, arr, low, high):
        if low < high:
            pi = self.partition(arr, low, high)
            self.quick_sort(arr, low, pi - 1)
            self.quick_sort(arr, pi + 1, high)

    def partition(self, arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            self.steps += 1  # comparison
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                self.steps += 3  # swap steps
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        self.steps += 3  # final swap steps
        return i + 1


class HeapSort(SortAlgorithm):
    def sort(self, data):
        self.steps = 0
        n = len(data)

        # Build a maxheap.
        for i in range(n // 2 - 1, -1, -1):
            self.heapify(data, n, i)

        # One by one extract elements
        for i in range(n - 1, 0, -1):
            data[i], data[0] = data[0], data[i]
            self.steps += 3  # swap steps
            self.heapify(data, i, 0)

        return self.steps

    def heapify(self, arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        # Compare left child
        if l < n:
            self.steps += 1  # comparison
            if arr[l] > arr[largest]:
                largest = l

        # Compare right child
        if r < n:
            self.steps += 1  # comparison
            if arr[r] > arr[largest]:
                largest = r

        # Swap if largest is not root
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.steps += 3  # swap steps
            self.heapify(arr, n, largest)


# Dictionary of available algorithms
ALGORITHMS = {
    "insertion": InsertionSort(),
    "merge": MergeSort(),
    "bubble": BubbleSort(),
    "quick": QuickSort(),
    "heap": HeapSort()
}


def read_data_from_csv(filename):
    df = pd.read_csv(filename, header=None)
    # Flatten and convert to list if multiple columns:
    data = df.values.flatten().tolist()
    return data


def generate_random_data(size, min_val=0, max_val=1000):
    return [random.randint(min_val, max_val) for _ in range(size)]


def compare_algorithms(data, algorithm_names):

    results = {}
    for name in algorithm_names:
        alg = ALGORITHMS[name]
        data_copy = data[:]
        steps = alg.sort(data_copy)
        results[name] = steps

    # Plot results as a bar chart
    plt.bar(results.keys(), results.values())
    plt.xlabel("Algorithms")
    plt.ylabel("Steps")
    plt.title("Algorithm Comparison")
    plt.show()


def compare_algorithm_with_complexity(algorithm_name, sizes, complexity="nlogn"):

    alg = ALGORITHMS[algorithm_name]
    x_vals = []
    y_vals = []
    for s in sizes:
        data = generate_random_data(s)
        steps = alg.sort(data)
        x_vals.append(s)
        y_vals.append(steps)

    # Compute theoretical complexity line
    # We'll just pick a scale factor so they appear in a comparable range.
    # For demonstration:
    # If complexity = nlogn: f(n) = n * log2(n)
    # If complexity = n2: f(n) = n^2
    # If complexity = n: f(n) = n
    # We'll scale them by a small factor c so they're visible on the same chart.

    c = 1
    if complexity == "n":
        theoretical = [c * n for n in x_vals]
    elif complexity == "n2":
        theoretical = [c * n * n for n in x_vals]
    elif complexity == "nlogn":
        theoretical = [c * n * math.log2(n) for n in x_vals]
    else:
        theoretical = [c * n for n in x_vals]  # default

    plt.plot(x_vals, y_vals, marker='o', label='Actual Steps')
    plt.plot(x_vals, theoretical, marker='x', label=f'Theoretical {complexity.upper()}')
    plt.xlabel("Input Size (n)")
    plt.ylabel("Steps")
    plt.title(f"{algorithm_name.capitalize()} Sort: Actual vs. {complexity.upper()}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage:

    # 1. Compare algorithms on a single dataset
    data = generate_random_data(1000, 0, 5000)
    compare_algorithms(data, ["insertion", "merge", "bubble", "quick", "heap"])

    # 2. Compare a single algorithm against its theoretical complexity
    # Using input sizes: 1000, 2000, 3000, 4000, 5000
    sizes = [1000, 2000, 3000, 4000, 5000]
    compare_algorithm_with_complexity("insertion", sizes, complexity="n2")