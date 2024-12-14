from project import read_data_from_file, generate_random_data, compare_algorithms, compare_algorithm_with_asymptotic_efficiency, ALGORITHMS
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def update_algorithm_selection():
    global chart_type_frame

    # Clear previous widgets
    for widget in algorithms_frame.winfo_children():
        widget.destroy()

    if compare_type_var.get() == "algorithms":
        # Create checkboxes for algorithms
        for alg in algorithms_list:
            alg_var[alg] = tk.BooleanVar()
            tk.Checkbutton(
                algorithms_frame, text=alg.capitalize(), variable=alg_var[alg]
            ).pack(side="left", padx=5)

        # Create and pack chart type frame
        chart_type_frame = tk.Frame(algorithms_frame)
        chart_type_frame.pack(pady=10, anchor="w")

        # Add label and radio buttons for chart type selection
        tk.Label(chart_type_frame, text="Chart Type (for comparing algorithms)").pack(
            side="left", padx=10
        )

        # Initialize chart type radio buttons
        growth_radio = tk.Radiobutton(
            chart_type_frame, text="Growth", variable=chart_type_var, value="growth"
        )
        growth_radio.pack(side="left", padx=5)

        nsteps_radio = tk.Radiobutton(
            chart_type_frame, text="N Steps", variable=chart_type_var, value="nsteps"
        )
        nsteps_radio.pack(side="left", padx=5)
    else:

        # Create radio buttons for algorithms
        for alg in algorithms_list:
            tk.Radiobutton(
                algorithms_frame,
                text=alg.capitalize(),
                variable=selected_algorithm,
                value=alg,
            ).pack(side="left", padx=5)



# File selection
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Data files", "*.csv *.xls *.xlsx")])
    file_label.config(text=file_path)


# Run comparison
def run_comparison():
    print("Running comparison")
    try:
        if use_file_var.get() and not file_label.cget("text"):
            messagebox.showerror("Error", "Please select a data file.")
            return

        size = int(size_entry.get()) if not use_file_var.get() else None
        data = read_data_from_file(file_label.cget("text")) if use_file_var.get() else generate_random_data(size)

        if compare_type_var.get() == "algorithms":
            selected_algs = [alg for alg in algorithms_list if alg_var[alg].get()]
            if len(selected_algs) < 2:
                messagebox.showerror("Error", "Please select at least two algorithms.")
                return
            compare_algorithms(data, selected_algs, chart_type_var.get(), gui_root=root)

        elif compare_type_var.get() == "asymptotic":
            if not selected_algorithm.get():
                messagebox.showerror("Error", "Please select an algorithm.", gui_root=root)
                return
            compare_algorithm_with_asymptotic_efficiency(data, selected_algorithm.get(), gui_root=root)
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid data size.")


# Main Window
root = tk.Tk()
root.title("Algorithm Comparison Tool")
root.geometry("500x500")

# Data Source Selection
use_file_var = tk.BooleanVar()
use_file_check = tk.Checkbutton(root, text="Use File", variable=use_file_var)
use_file_check.pack(pady=5)

file_label = tk.Label(root, text="No file selected")
file_label.pack(pady=5)

browse_btn = tk.Button(root, text="Browse", command=open_file)
browse_btn.pack(pady=5)

tk.Label(root, text="Or Enter Data Size").pack(pady=5)
size_entry = tk.Entry(root)
size_entry.pack(pady=5)

# Comparison Type
compare_type_var = tk.StringVar(value="algorithms")
tk.Label(root, text="Comparison Type").pack(pady=5)
tk.Radiobutton(root, text="Compare Algorithms", variable=compare_type_var, value="algorithms",
               command=update_algorithm_selection).pack()
tk.Radiobutton(root, text="Compare with Asymptotic Efficiency", variable=compare_type_var, value="asymptotic",
               command=update_algorithm_selection).pack()

# Algorithm Selection
algorithms_list = ALGORITHMS.keys()
alg_var = {}
selected_algorithm = tk.StringVar(value="insertion")


algorithms_frame = tk.Frame(root)
algorithms_frame.pack(pady=10)

chart_type_var = tk.StringVar(value="growth")

# Run Button
run_btn = tk.Button(root, text="Run Comparison", command=run_comparison)
run_btn.pack(pady=20)

# Initialize GUI
update_algorithm_selection()
root.mainloop()