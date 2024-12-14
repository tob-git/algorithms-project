import unittest
from unittest.mock import patch
from project import SelectionSort, InsertionSort, BubbleSort, ShellSort, MergeSort, QuickSort, HeapSort, validate_choice, validate_positive_int, validate_filename, validate_algorithms
class TestSortingAlgorithms(unittest.TestCase):

    def setUp(self):
        self.random_data = [3, 1, 4, 5, 2]
        self.expected_result = [1, 2, 3, 4, 5]

    def test_selection_sort(self):
        sorter = SelectionSort()
        data = self.random_data.copy()
        sorter.sort(data)
        self.assertEqual(data, self.expected_result)

    def test_insertion_sort(self):
        sorter = InsertionSort()
        data = self.random_data.copy()
        sorter.sort(data)
        self.assertEqual(data, self.expected_result)

    def test_bubble_sort(self):
        sorter = BubbleSort()
        data = self.random_data.copy()
        sorter.sort(data)
        self.assertEqual(data, self.expected_result)

    def test_shell_sort(self):
        sorter = ShellSort()
        data = self.random_data.copy()
        sorter.sort(data)
        self.assertEqual(data, self.expected_result)

    def test_merge_sort(self):
        sorter = MergeSort()
        data = self.random_data.copy()
        sorter.sort(data)
        self.assertEqual(data, self.expected_result)

    def test_quick_sort(self):
        sorter = QuickSort()
        data = self.random_data.copy()
        sorter.sort(data)
        self.assertEqual(data, self.expected_result)

    def test_heap_sort(self):
        sorter = HeapSort()
        data = self.random_data.copy()
        sorter.sort(data)
        self.assertEqual(data, self.expected_result)

    def test_edge_case_empty_list(self):
        sorter = SelectionSort()
        data = []
        sorter.sort(data)
        self.assertEqual(data, [])

    def test_edge_case_single_element(self):
        sorter = SelectionSort()
        data = [42]
        sorter.sort(data)
        self.assertEqual(data, [42])

    @patch('builtins.input', side_effect=['option1'])
    def test_validate_choice_valid(self, mock_input):
        result = validate_choice("Choose an option: ", ['option1', 'option2'])
        self.assertEqual(result, 'option1')

    @patch('builtins.input', side_effect=['invalid', 'option2'])
    def test_validate_choice_invalid_then_valid(self, mock_input):
        result = validate_choice("Choose an option: ", ['option1', 'option2'])
        self.assertEqual(result, 'option2')

    @patch('builtins.input', side_effect=['5'])
    def test_validate_positive_int_valid(self, mock_input):
        result = validate_positive_int("Enter a positive integer: ")
        self.assertEqual(result, 5)

    @patch('builtins.input', side_effect=['-3', 'abc', '7'])
    def test_validate_positive_int_invalid_then_valid(self, mock_input):
        result = validate_positive_int("Enter a positive integer: ")
        self.assertEqual(result, 7)

    @patch('builtins.input', side_effect=['my_file.txt'])
    def test_validate_filename_valid(self, mock_input):
        result = validate_filename("Enter a filename: ")
        self.assertEqual(result, 'my_file.txt')

    @patch('builtins.input', side_effect=['', 'test_file.py'])
    def test_validate_filename_empty_then_valid(self, mock_input):
        result = validate_filename("Enter a filename: ")
        self.assertEqual(result, 'test_file.py')

    @patch('builtins.input', side_effect=['alg1, alg2'])
    def test_validate_algorithms_valid(self, mock_input):
        result = validate_algorithms(
            "Choose algorithms: ",
            ['alg1', 'alg2', 'alg3'],
            min_count=1
        )
        self.assertEqual(result, ['alg1', 'alg2'])

    @patch('builtins.input', side_effect=['alg4', 'alg1, alg3'])
    def test_validate_algorithms_invalid_then_valid(self, mock_input):
        result = validate_algorithms(
            "Choose algorithms: ",
            ['alg1', 'alg2', 'alg3'],
            min_count=2
        )
        self.assertEqual(result, ['alg1', 'alg3'])

    @patch('builtins.input', side_effect=['alg1, alg2, alg3'])
    def test_validate_algorithms_exact_count(self, mock_input):
        result = validate_algorithms(
            "Choose algorithms: ",
            ['alg1', 'alg2', 'alg3'],
            exact_count=3
        )
        self.assertEqual(result, ['alg1', 'alg2', 'alg3'])

    @patch('builtins.input', side_effect=['alg1, alg2', 'alg1, alg2, alg3'])
    def test_validate_algorithms_exact_count_retry(self, mock_input):
        result = validate_algorithms(
            "Choose algorithms: ",
            ['alg1', 'alg2', 'alg3'],
            exact_count=3
        )
        self.assertEqual(result, ['alg1', 'alg2', 'alg3'])

if __name__ == "__main__":
    unittest.main()