# main_gui.py
import openpyxl

import dearpygui.dearpygui as dpg
from funcs import (
    read_data_from_file,
    generate_random_data,
    compare_algorithms,
    compare_algorithm_with_asymptotic_efficiency,
    ALGORITHMS,
)
import math

# Initialize DearPyGui context
dpg.create_context()

# Global Variables
selected_file = ""
use_file = False
compare_type = "algorithms"
selected_algorithms = {}
selected_algorithm = "insertion"

# Helper Functions
def update_algorithm_selection():
    global compare_type
    compare_type = dpg.get_value("comparison_type")
    print(compare_type)
    # Clear existing algorithm selection widgets
    dpg.delete_item("algorithm_selection_group", children_only=True)
    print("1")
    if compare_type == "algorithms":
        # Create checkboxes for algorithms
        with dpg.group(horizontal=True, parent="algorithm_selection_group"):
            for alg in ALGORITHMS.keys():
                dpg.add_checkbox(
                    label=alg.capitalize(),
                    tag=f"alg_checkbox_{alg}",
                    default_value=False,
                    callback=update_selected_algorithms
                )

    else:
        print("2")
        with dpg.group(horizontal=True, parent="algorithm_selection_group"):
            dpg.add_radio_button(
                items=[alg.capitalize() for alg in ALGORITHMS.keys()],
                tag="single_algorithm",
                callback=lambda s, a, u: set_selected_algorithm(a)
            )



def set_selected_algorithm(value):
    global selected_algorithm
    selected_algorithm = value.lower()

def update_selected_algorithms(sender, app_data, user_data):
    global selected_algorithms
    selected_algorithms = {
        alg: dpg.get_value(f"alg_checkbox_{alg}") for alg in ALGORITHMS.keys()
    }



def file_selected(sender, app_data, user_data):
    global selected_file
    if app_data["file_path_name"]:
        selected_file = app_data["file_path_name"]
        dpg.set_value("file_label", selected_file)

def run_comparison(sender, app_data, user_data):
    try:
        # Data Handling
        if dpg.get_value("use_file_checkbox"):
            if not selected_file:
                dpg.show_item("error_no_file")
                return
            print(selected_file)
            data = read_data_from_file(selected_file)
        else:
            size_str = dpg.get_value("size_input")
            if not size_str.isdigit():
                print('XXX')
                dpg.show_item("error_invalid_size")
                return
            size = int(size_str)
            data = generate_random_data(size)

        # Comparison Handling
        if compare_type == "algorithms":
            selected_algs = [alg for alg, val in selected_algorithms.items() if val]
            if len(selected_algs) < 2:
                dpg.show_item("error_select_two_algorithms")
                return
            sizes, comparison_data = compare_algorithms(data, selected_algs)
            plot_comparison_algorithms_growth(sizes, comparison_data)
        else:
            if not selected_algorithm:
                dpg.show_item("error_select_algorithm")
                return
            sizes, comparison_data = compare_algorithm_with_asymptotic_efficiency(
                data, selected_algorithm
            )
            plot_comparison_algorithms_growth(sizes, comparison_data)

    except ValueError:
        print("ValueError")
        dpg.show_item("error_invalid_size")
    except TypeError:
        print("TypeError")
        dpg.set_value("error_message", "something went wrong with your file make sure its numbers only")
        dpg.show_item("error_general")
    except Exception as e:
        dpg.set_value("error_message", str(e))
        dpg.show_item("error_general")

def plot_comparison_algorithms_growth(sizes, comparison_data):
    # Clear previous plot
    if dpg.does_item_exist("comparison_plot"):
        dpg.delete_item("comparison_plot")
    with dpg.plot(label="Algorithm Growth Comparison", width=1420, height=630, tag="comparison_plot", parent="Algorithm Comparison Tool"):
        dpg.add_plot_legend()
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Input Size", parent="comparison_plot")
        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Number of Steps", parent="comparison_plot")
        print("2")
        for alg, steps in comparison_data.items():
            print("3")
            dpg.add_line_series(
                x=sizes,
                y=steps,
                label=alg.capitalize(),
                parent=y_axis
            )
        print("4")

def show_file_dialog(sender, app_data, user_data):
    print("show_file_dialog")
    dpg.show_item("file_dialog")

def open_file_dialog(sender, app_data, user_data):
    print(f"File Selected: {app_data['file_path_name']}")
# Window Setup
with dpg.window(label="Algorithm Comparison Tool", width=600, height=700, tag="Algorithm Comparison Tool"):
    # Data Source Selection
    with dpg.group(horizontal=True):
        dpg.add_checkbox(label="Use File", tag="use_file_checkbox", default_value=False)
        dpg.add_button(label="Browse", callback=show_file_dialog)
    dpg.add_text("No file selected", tag="file_label")






    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=file_selected,
        tag="file_dialog",
        width=800,  # Set width of the file dialog
        height=600  # Set height of the file dialog
    ):
        # Add as many file extension filters as needed
        dpg.add_file_extension(".csv", color=(0, 255, 0, 255))
        dpg.add_file_extension(".xls", color=(0, 255, 0, 255))
        dpg.add_file_extension(".xlsx", color=(0, 255, 0, 255))

    dpg.add_separator()

    dpg.add_text("Or Enter Data Size:")
    dpg.add_input_text(label="Data Size", tag="size_input", default_value="1000", width=50)

    dpg.add_separator()

    # Comparison Type
    dpg.add_text("Comparison Type:")
    with dpg.group(horizontal=True):
        dpg.add_radio_button(
            items=["algorithms", "asymptotic"],
            tag="comparison_type",
            default_value="algorithms",
            callback=update_algorithm_selection
        )

    # Algorithm Selection
    with dpg.child_window(height=50, tag="algorithm_selection_group"):
        update_algorithm_selection()  # Initialize selection

    dpg.add_separator()

    # Run Button
    dpg.add_button(label="Run Comparison", callback=run_comparison)

    # Error Messages
    with dpg.window(label="No File Selected", modal=True, show=False, tag="error_no_file"):
        dpg.add_text("Please select a data file.")
        dpg.add_spacer(height=5)
        dpg.add_button(label="OK", callback=lambda: dpg.hide_item("error_no_file"))

    with dpg.window(label="Invalid Size", modal=True, show=False, tag="error_invalid_size"):
        dpg.add_text("Please enter a valid data size.")
        dpg.add_spacer(height=5)
        dpg.add_button(label="OK", callback=lambda: dpg.hide_item("error_invalid_size"))

    # --- Error Dialog: Select Two Algorithms ---
    with dpg.window(label="Select Two Algorithms", modal=True, show=False, tag="error_select_two_algorithms"):
        dpg.add_text("Please select at least two algorithms.")
        dpg.add_spacer(height=5)
        dpg.add_button(label="OK", callback=lambda: dpg.hide_item("error_select_two_algorithms"))

    # --- Error Dialog: Select One Algorithm ---
    with dpg.window(label="Select an Algorithm", modal=True, show=False, tag="error_select_algorithm"):
        dpg.add_text("Please select an algorithm.")
        dpg.add_spacer(height=5)
        dpg.add_button(label="OK", callback=lambda: dpg.hide_item("error_select_algorithm"))

    # --- Error Dialog: General Error ---
    with dpg.window(label="General Error", modal=True, show=False, tag="error_general"):
        dpg.add_text("An unexpected error occurred.")
        dpg.add_text("", tag="error_message")  # for extra info
        dpg.add_spacer(height=5)
        dpg.add_button(label="OK", callback=lambda: dpg.hide_item("error_general"))

# Show the DearPyGui viewport
dpg.create_viewport(title='Algorithm Comparison Tool', width=600, height=700)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Algorithm Comparison Tool", True)
dpg.start_dearpygui()
dpg.destroy_context()
