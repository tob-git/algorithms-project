import random
import math
import matplotlib.pyplot as plt
import pandas as pd
from ui import display_plot


# Common interface for sorting algorithms
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
        self.O = "O(n^2)"
        self.Theta = "Θ(n^2)"
        self.Omega = "Ω(n^2)"

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
        self.O = "O(n^2)"
        self.Theta = "Θ(n log(n)^2)"
        self.Omega = "Ω(n log(n))"

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

class CountingSort(SortAlgorithm):

    def __init__(self):
        super().__init__()
        self.O = "O(n + k)"
        self.Theta = "Θ(n + k)"
        self.Omega = "Ω(n + k)"

    def sort(self, data):
        steps = 0
        if not data:
            return steps

        max_val = max(data)
        steps += len(data)  # for max computation

        count = [0] * (max_val + 1)
        steps += max_val + 1  # initialization

        for num in data:
            count[num] += 1
            steps += 1  # counting

        output_idx = 0
        for i in range(len(count)):
            while count[i] > 0:
                data[output_idx] = i
                count[i] -= 1
                output_idx += 1
                steps += 2  # assignment + count decrement

        return steps

class RadixSort(SortAlgorithm):
    def __init__(self):
        super().__init__()
        self.O = "O(nk)"
        self.Theta = "Θ(nk)"
        self.Omega = "Ω(nk)"

    def counting_sort(self, data, exp, steps):
        n = len(data)
        output = [0] * n
        count = [0] * 10
        steps += 10  # initialization

        for i in range(n):
            index = (data[i] // exp) % 10
            count[index] += 1
            steps += 2  # calculation + count increment

        for i in range(1, 10):
            count[i] += count[i - 1]
            steps += 1  # cumulative sum

        for i in range(n - 1, -1, -1):
            index = (data[i] // exp) % 10
            output[count[index] - 1] = data[i]
            count[index] -= 1
            steps += 3  # calculations and assignments

        for i in range(n):
            data[i] = output[i]
            steps += 1  # final assignment

        return steps

    def sort(self, data):
        steps = 0
        if not data:
            return steps

        max_val = max(data)
        steps += len(data)  # finding max

        exp = 1
        while max_val // exp > 0:
            steps = self.counting_sort(data, exp, steps)
            exp *= 10
            steps += 1  # exp update

        return steps

class InsertionSort(SortAlgorithm):
    def __init__(self):
        super().__init__()
        self.O = "O(n^2)"
        self.Theta = "Θ(n^2)"
        self.Omega = "Ω(n)"

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
        self.O = "O(n log n)"
        self.Theta = "Θ(n log n)"
        self.Omega = "Ω(n log n)"

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
        self.O = "O(n^2)"
        self.Theta = "Θ(n^2)"
        self.Omega = "Ω(n)"

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
        self.O = "O(n^2)"
        self.Theta = "Θ(n log n)"
        self.Omega = "Ω(n log n)"

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
        self.O = "O(n log n)"
        self.Theta = "Θ(n log n)"
        self.Omega = "Ω(n log n)"

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
    "shell": ShellSort(),
    "counting": CountingSort(),
    "radix": RadixSort()
}


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


def generate_random_data(size):
    max_val = max(1000, size * 10)
    return [random.randint(0, max_val) for _ in range(size)]




def compare_algorithms(data, algorithm_names, chart_type="growth", split_points=5):
    n = len(data)

    if chart_type == "growth":
        # Split data sizes for growth graph
        sizes = [int(n * i / split_points) for i in range(1, split_points + 1)]
        results = {name: [] for name in algorithm_names}

        # Collect performance data for each algorithm at different sizes
        for size in sizes:
            for name in algorithm_names:
                alg = ALGORITHMS[name]
                data_sample = data[:size]
                steps = alg.sort(data_sample)
                results[name].append(steps)

        # Plot the growth graph
        for name, steps in results.items():
            plt.plot(sizes, steps, marker='o', label=name)
        plt.xlabel("Data Size (n)")
        plt.ylabel("Steps")
        plt.title("Algorithm Comparison (growth)")

    elif chart_type == "nsteps":
        # Collect total performance data
        results = {}
        for name in algorithm_names:
            alg = ALGORITHMS[name]
            steps = alg.sort(data[:])
            results[name] = steps

        # Plot the bar chart
        plt.bar(results.keys(), results.values())
        plt.xlabel("Algorithms")
        plt.ylabel("Total Steps")
        plt.title("Algorithm Comparison (n of steps)")



    plt.legend()
    plt.show()

def compare_algorithm_with_asymptotic_efficiency(data, algorithm_name, split_points=5):
    n = len(data)
    sizes = [int(n * i / split_points) for i in range(1, split_points + 1)]

    alg = ALGORITHMS[algorithm_name]
    x_vals = []
    y_vals = []

    # Collect actual steps
    for s in sizes:
        data_sample = data[:s]
        steps = alg.sort(data_sample)
        x_vals.append(s)
        y_vals.append(steps)

    # Compute theoretical complexities
    complexities = {
        "Worst Case": alg.O,
        "Average Case": alg.Theta,
        "Best Case": alg.Omega
    }

    theoretical = {case: [] for case in complexities}

    # Calculate theoretical values based on complexity functions
    for case, complexity in complexities.items():
        for n in x_vals:
            if complexity == "O(n)":
                theoretical[case].append(n)
            elif complexity == "O(n^2)":
                theoretical[case].append(n ** 2)
            elif complexity == "O(n log n)":
                theoretical[case].append(n * math.log2(n))
            elif complexity == "O(1)":
                theoretical[case].append(1)
            else:
                theoretical[case].append(n)  # Default for unexpected cases

    # Plot the results
    plt.plot(x_vals, y_vals, marker='o', label='Actual Steps')

    for case, values in theoretical.items():
        plt.plot(x_vals, values, marker='x', label=f'{case} ({complexities[case]})')

    plt.xlabel("Input Size (n)")
    plt.ylabel("Steps")
    plt.title(f"{algorithm_name.capitalize()} Sort: Actual vs. Theoretical Complexity")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Select operation
    print("Choose the operation:")
    print("1. Compare algorithms on a single dataset")
    print("2. Compare a single algorithm against its theoretical complexity")
    choice = input("Enter your choice (1 or 2): ")

    # Select data source
    print("Choose the data source:")
    print("1. Generate random data")
    print("2. Load data from file")
    data_choice = input("Enter your choice (1 or 2): ")

    if data_choice == "1":
        size = int(input("Enter the size of the data: "))
        data = generate_random_data(size)
    else:
        filename = input("Enter the file name (with extension): ")
        data = read_data_from_file(filename)

    # Execute selected operation
    if choice == "1":
        print("Available algorithms:", ", ".join(ALGORITHMS.keys()))
        algorithms = input("Enter the algorithms separated by commas (e.g., insertion,merge,quick): ").split(",")
        algorithms = [algo.strip().lower() for algo in algorithms]
        print("Chart Type Options:")
        print("1 - Growth")
        print("2 - Number of Steps (nsteps)")

        # Get user input
        user_input = input("Enter chart type (1 for Growth, 2 for Number of Steps): ").strip()

        # Map input to chart types
        if user_input == "1":
            chart_type = "growth"
        elif user_input == "2":
            chart_type = "nsteps"
        else:
            chart_type = None
            print("Invalid selection. Please enter 1 or 2.")
        compare_algorithms(data, algorithms, chart_type)
    elif choice == "2":
        print("Available algorithms:", ", ".join(ALGORITHMS.keys()))
        algorithm = input("Enter the algorithm (e.g., insertion, merge, quick): ")
        compare_algorithm_with_asymptotic_efficiency(data, algorithm)
    else:
        print("Invalid choice!")
