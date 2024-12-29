# project.py

import math
import pandas as pd
import random

class SortAlgorithm:
    def __init__(self):
        self.O = ""        # Worst-case complexity
        self.Theta = ""    # Average-case complexity
        self.Omega = ""    # Best-case complexity

    def sort(self, data):
        raise NotImplementedError

class SelectionSort(SortAlgorithm):
    def __init__(self):
        super().__init__()
        self.O = "(n^2)"
        self.Theta = "(n^2)"
        self.Omega = "(n^2)"

    def sort(self, data):
        steps = 0
        n = len(data)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                steps += 1  # comparison
                if data[j] < data[min_idx]:
                    min_idx = j
            if min_idx != i:
                data[i], data[min_idx] = data[min_idx], data[i]
                steps += 3  # swap (read/write)
        return steps

class ShellSort(SortAlgorithm):
    def __init__(self):
        super().__init__()
        self.O = "(n^2)"
        self.Theta = "(n log(n)^2)"
        self.Omega = "(n log(n))"

    def sort(self, data):
        steps = 0
        n = len(data)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                temp = data[i]
                j = i
                steps += 1  # initial assignment
                while j >= gap and data[j - gap] > temp:
                    steps += 2  # comparison + assignment
                    data[j] = data[j - gap]
                    j -= gap
                data[j] = temp
                steps += 1  # final assignment
            gap //= 2
        return steps

class InsertionSort(SortAlgorithm):
    def __init__(self):
        super().__init__()
        self.O = "(n^2)"
        self.Theta = "(n^2)"
        self.Omega = "(n)"

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
    def __init__(self):
        super().__init__()
        self.O = "(n log n)"
        self.Theta = "(n log n)"
        self.Omega = "(n log n)"

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
    def __init__(self):
        super().__init__()
        self.O = "(n^2)"
        self.Theta = "(n^2)"
        self.Omega = "(n)"

    def sort(self, data):
        steps = 0
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                steps += 1  # comparison
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
                    steps += 3  # two assignments for swap
        return steps

class QuickSort(SortAlgorithm):
    def __init__(self):
        super().__init__()
        self.O = "(n^2)"
        self.Theta = "(n log n)"
        self.Omega = "(n log n)"

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
    def __init__(self):
        super().__init__()
        self.O = "(n log n)"
        self.Theta = "(n log n)"
        self.Omega = "(n log n)"

    def sort(self, data):
        self.steps = 0
        n = len(data)

        # Build a maxheap
        for i in range(n // 2 - 1, -1, -1):
            self.heapify(data, n, i)

        # Extract elements one by one
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
    "heap": HeapSort(),
    "selection": SelectionSort(),
    "shell": ShellSort()
}

def compare_algorithms(data, selected_algs, split_points=100):
    n = len(data)


    # Split data sizes for growth graph
    sizes = [int(n * i / split_points) for i in range(1, split_points + 1)]
    results = {name: [] for name in selected_algs}

    # Collect performance data for each algorithm at different sizes
    for size in sizes:
        for name in selected_algs:
            alg = ALGORITHMS[name]
            # As the data is a list of integers, we don't need to deep copy it
            data_sample = data[:size]
            steps = alg.sort(data_sample)
            results[name].append(steps)
    return sizes, results


def compare_algorithm_with_asymptotic_efficiency(data, algorithm_name, split_points=20):
    import math  # Ensure math is imported
    n = len(data)

    # Split data sizes for growth graph
    sizes = [int(n * i / split_points) for i in range(1, split_points + 1)]

    # Get the selected algorithm
    alg = ALGORITHMS[algorithm_name]

    # Initialize results dictionary for actual performance
    results = {"Actual": []}

    # Collect actual steps
    for size in sizes:
        data_sample = data[:size]
        steps = alg.sort(data_sample)
        results["Actual"].append(steps)

    # Initialize results dictionary for theoretical values
    complexities = {
        "Worst Case": alg.O,
        "Average Case": alg.Theta,
        "Best Case": alg.Omega
    }

    for case, complexity in complexities.items():
        theoretical_result = []
        for size in sizes:
            if complexity == "(n)":
                theoretical_result.append(size)
            elif complexity == "(n^2)":
                theoretical_result.append(size ** 2)
            elif complexity == "(n log n)":
                theoretical_result.append(size * math.log2(size) if size > 0 else 0)
            elif complexity == "(1)":
                theoretical_result.append(1)
            else:
                theoretical_result.append(size)  # Default for unexpected cases

        # Add theoretical result to the dictionary with the complexity label
        results[f"{case}: {complexity}"] = theoretical_result

    return sizes, results

def read_data_from_file(filename):

    # Determine file type based on extension
    if filename.lower().endswith('.csv'):
        df = pd.read_csv(filename, header=None)
    elif filename.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filename, header=None)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx/.xls files.")

    # Flatten and convert to list if multiple columns
    data = df.values.flatten().tolist()
    return data


def generate_random_data(size, filename='random_nums.xlsx'):
    max_val = max(1000, size * 10)
    data = [random.randint(0, max_val) for _ in range(size)]
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False, header=False)
    print(f"Data saved to {filename}")
    return data