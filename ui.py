from project import read_data_from_file, generate_random_data, compare_algorithms, compare_algorithm_with_asymptotic_efficiency
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def display_plot(fig):
    global canvas
    if canvas:
        canvas.get_tk_widget().pack_forget()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Setup the Tkinter GUI
root = tk.Tk()
root.title("Algorithm Comparison Tool")
root.geometry("800x600")

# Add a Frame for the Plot
plot_frame = ttk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Test the Plot Function
data = list(range(1, 1000))
compare_algorithms(data, ["insertion", "merge", "quick"], chart_type="growth")

def update_algorithm_selection():
    for widget in algorithms_frame.winfo_children():
        widget.destroy()

    if compare_type_var.get() == "algorithms":
        for alg in algorithms_list:
            alg_var[alg] = tk.BooleanVar()
            tk.Checkbutton(algorithms_frame, text=alg.capitalize(), variable=alg_var[alg]).pack(anchor="w")
        chart_type_label.pack(pady=5)
        growth_radio.pack(anchor="w")
        nsteps_radio.pack(anchor="w")
    else:
        selected_algorithm.set("insertion")
        for alg in algorithms_list:
            tk.Radiobutton(algorithms_frame, text=alg.capitalize(), variable=selected_algorithm, value=alg).pack(
                anchor="w")
        chart_type_label.pack_forget()
        growth_radio.pack_forget()
        nsteps_radio.pack_forget()


# File selection
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Data files", "*.csv *.xls *.xlsx")])
    file_label.config(text=file_path)


# Run comparison
def run_comparison():
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
            compare_algorithms(data, selected_algs, chart_type_var.get())

        elif compare_type_var.get() == "asymptotic":
            if not selected_algorithm.get():
                messagebox.showerror("Error", "Please select an algorithm.")
                return
            compare_algorithm_with_asymptotic_efficiency(data, selected_algorithm.get())
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
algorithms_list = ["insertion", "merge", "quick", "bubble", "heap"]
alg_var = {}
selected_algorithm = tk.StringVar()

algorithms_frame = tk.Frame(root)
algorithms_frame.pack(pady=10)

# Chart Type
chart_type_label = tk.Label(root, text="Chart Type (for comparing algorithms)")
chart_type_label.pack(pady=5)

chart_type_var = tk.StringVar(value="growth")
growth_radio = tk.Radiobutton(root, text="Growth", variable=chart_type_var, value="growth")
growth_radio.pack(anchor="w")

nsteps_radio = tk.Radiobutton(root, text="N Steps", variable=chart_type_var, value="nsteps")
nsteps_radio.pack(anchor="w")

# Run Button
run_btn = tk.Button(root, text="Run Comparison", command=run_comparison)
run_btn.pack(pady=20)

# Initialize GUI
update_algorithm_selection()
root.mainloop()