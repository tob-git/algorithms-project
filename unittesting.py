import unittest

from project import SelectionSort, InsertionSort, BubbleSort, ShellSort, MergeSort, QuickSort, HeapSort
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

if __name__ == "__main__":
    unittest.main()