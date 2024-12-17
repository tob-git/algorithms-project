import random
import math
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
current_canvas = None
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
def compare_algorithms(data, selected_algs, chart_type="growth", split_points=20, gui_root=None):
    n = len(data)
    is_gui = gui_root is not None
    global current_canvas

    # Create figure and axis
    if is_gui:
        # Clear the old canvas if it exists
        if current_canvas is not None:
            current_canvas.get_tk_widget().pack_forget()
            current_canvas.get_tk_widget().destroy()
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    if chart_type == "growth":
        # Split data sizes for growth graph
        sizes = [int(n * i / split_points) for i in range(1, split_points + 1)]
        results = {name: [] for name in selected_algs}

        # Collect performance data for each algorithm at different sizes
        for size in sizes:
            for name in selected_algs:
                alg = ALGORITHMS[name]
                #as the data is a list of integers, we don't need to deep copy it
                data_sample = data[:size]
                steps = alg.sort(data_sample)
                results[name].append(steps)

        # Plot the growth graph
        styles = ['-', '--', '-.', ':', (0, (5, 10)), (0, (1, 1)), (0, (3, 5, 1, 5))]
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
        markers = ['o', 's', 'D', '^', 'v', 'x', '*']
        alpha_value = 0.7

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, (name, steps) in enumerate(results.items()):
            ax.plot(
                sizes, steps,
                linestyle=styles[idx % len(styles)],
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                alpha=alpha_value,
                label=name
            )
        ax.set_xlabel("Data Size (n)")
        ax.set_ylabel("Steps")
        ax.set_title("Algorithm Comparison (growth)")
        ax.legend()

    elif chart_type == "nsteps":
        # Collect total performance data
        results = {}
        for name in selected_algs:
            alg = ALGORITHMS[name]
            steps = alg.sort(data[:])
            results[name] = steps

        # Plot the bar chart
        ax.bar(results.keys(), results.values())
        ax.set_xlabel("Algorithms")
        ax.set_ylabel("Total Steps")
        ax.set_title("Algorithm Comparison (n of steps)")

    # Display the plot
    if is_gui:
        # Embed the new plot in Tkinter
        current_canvas = FigureCanvasTkAgg(fig, master=gui_root)
        current_canvas.draw()
        current_canvas.get_tk_widget().pack(pady=10, expand=True, fill="both")
    else:
        plt.show()

def compare_algorithm_with_asymptotic_efficiency(data, algorithm_name, split_points=20, gui_root=None):
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
            if complexity == "(n)":
                theoretical[case].append(n)
            elif complexity == "(n^2)":
                theoretical[case].append(n ** 2)
            elif complexity == "(n log n)":
                theoretical[case].append(n * math.log2(n))
            elif complexity == "(1)":
                theoretical[case].append(1)
            else:
                theoretical[case].append(n)  # Default for unexpected cases

    # Create figure and axis
    is_gui = gui_root is not None
    global current_canvas

    if is_gui:
        # Clear old canvas if exists
        if current_canvas is not None:
            current_canvas.get_tk_widget().pack_forget()
            current_canvas.get_tk_widget().destroy()
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Define a small epsilon value for offsets


    # Plot the actual steps
    ax.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='Actual Steps')

    # Use different styles and apply offsets for theoretical lines
    styles = ['--', '-.', ':']
    colors = ['r', 'g', 'm']
    alpha_value = 0.7

    # Apply offsets
    for idx, (case, values) in enumerate(theoretical.items()):

        ax.plot(x_vals, values, linestyle=styles[idx % len(styles)],
                color=colors[idx % len(colors)], alpha=alpha_value,
                marker='x', label=f'{case} ({complexities[case]})')

    ax.set_xlabel("Input Size (n)")
    ax.set_ylabel("Steps")
    ax.set_title(f"{algorithm_name.capitalize()} Sort: Actual vs. Theoretical Complexity")
    ax.legend()

    if is_gui:
        current_canvas = FigureCanvasTkAgg(fig, master=gui_root)
        current_canvas.draw()
        current_canvas.get_tk_widget().pack(pady=10)
    else:
        plt.show()

def validate_choice(prompt, options):
    while True:
        choice = input(prompt).strip()
        if choice in options:
            return choice
        print(f"Invalid input. Please choose from {', '.join(options)}.")

def validate_positive_int(prompt):
    while True:
        try:
            value = int(input(prompt).strip())
            if value > 0:
                return value
            print("The value must be a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")

def validate_filename(prompt):
    while True:
        filename = input(prompt).strip()
        if filename:
            return filename
        print("Filename cannot be empty.")

def validate_algorithms(prompt, available_algs, min_count=1, exact_count=None):
    while True:
        algs = input(prompt).strip().lower().split(",")
        selected_algs = [alg.strip() for alg in algs if alg.strip()]

        if exact_count and len(selected_algs) != exact_count:
            print(f"You must choose exactly {exact_count} algorithm(s).")
        elif  all(alg in available_algs for alg in selected_algs):
            if len(selected_algs) < min_count:
                print(f"You must choose at least {min_count} algorithm(s).")
            else:
                return selected_algs
        else:
            print(f"Invalid algorithm(s). Available algorithms are: {', '.join(available_algs)}.")


if __name__ == "__main__":
    print("Choose the operation:")
    print("1. Compare algorithms on a single dataset")
    print("2. Compare a single algorithm against its theoretical complexity")
    choice = validate_choice("Enter your choice (1 or 2): ", ["1", "2"])

    # Select data source
    print("Choose the data source:")
    print("1. Generate random data")
    print("2. Load data from file")
    data_choice = validate_choice("Enter your choice (1 or 2): ", ["1", "2"])

    if data_choice == "1":
        size = validate_positive_int("Enter the size of the data: ")
        data = generate_random_data(size)
    else:
        filename = validate_filename("Enter the file name (with extension): ")
        data = read_data_from_file(filename)

    # Execute selected operation
    if choice == "1":
        print("Available algorithms:", ", ".join(ALGORITHMS.keys()))
        selected_algs = validate_algorithms(
            "Enter at least two algorithms separated by commas (e.g., insertion,merge,quick): ",
            ALGORITHMS.keys(),
            min_count=2
        )

        print("Chart Type Options:")
        print("1 - Growth")
        print("2 - Number of Steps (nsteps)")
        chart_type_input = validate_choice("Enter chart type (1 for Growth, 2 for Number of Steps): ", ["1", "2"])

        chart_type = "growth" if chart_type_input == "1" else "nsteps"
        compare_algorithms(data, selected_algs, chart_type)

    elif choice == "2":
        print("Available algorithms:", ", ".join(ALGORITHMS.keys()))
        algorithm = validate_algorithms(
            "Enter exactly one algorithm: ",
            ALGORITHMS.keys(),
            exact_count=1
        )[0]
        compare_algorithm_with_asymptotic_efficiency(data, algorithm)