import unittest

import numpy as np

from wavelet import Wavelet


class WaveletShortTests(unittest.TestCase):
    def setUp(self):
        self.w = Wavelet(m=2, g=1)

    def test_mg(self):
        self.assertEqual(self.w.mg, 2)

    def test_allocate_matrix(self):
        self.assertEqual(self.w.A, None)
        ans = np.zeros((self.w.m, self.w.mg))
        self.w._allocate_a_matrix()
        self.assertTrue(np.array_equal(ans, self.w.A))

    def test_vstack(self):
        arr1 = np.array([0, 1])
        arr2 = np.array([2, 3])
        ret = np.vstack((arr1, arr2))
        ans = np.array([[0, 1], [2, 3]])
        self.assertTrue(np.array_equal(ret, ans))

    def test_get_raw_4_lines(self):
        file = 'mcw2_4.txt'
        l0, l1 = list(self.w._get_raw_lines(file))
        self.assertTrue(np.array_equal(l0, np.array([-1, 10])))
        self.assertTrue(np.array_equal(l1, np.array([111, -1.9])))

    def test_set_a_coefficients(self):
        self.w = Wavelet(m=2, g=1)
        file = 'mcw2_4.txt'
        ans = np.array([[-1, 10], [111, -1.9]])
        self.w._set_a_coefficients(file)
        self.assertTrue(np.array_equal(ans, self.w.A))

    def test_allocate_mcw(self):
        message_length = 4
        self.w._allocate_mcw(message_length)
        ans = np.zeros((self.w.m, message_length + self.w.mg - self.w.m))
        self.assertTrue(np.array_equal(self.w.mcw, ans))

    def test_mcw_from_coefficients(self):
        file = 'mcw2_4.txt'
        message_length = 4
        self.w.mcw_from_coefficients(file, message_length)
        mcw_lines, mcw_columns = self.w.mcw.shape
        self.assertEqual(self.w.m, mcw_lines)
        self.assertEqual(message_length + self.w.mg - self.w.m, mcw_columns)
        self.assertTrue(np.array_equal(self.w.mcw[:self.w.m, :self.w.mg], self.w.A))

    def test_mcw_displacement(self):
        file = 'mcw2_4.txt'
        message_length = 6
        self.w.mcw_from_coefficients(file, message_length)
        original_mcw = np.copy(self.w.mcw)
        # Displace by one m
        self.w.displace_mcw(1)
        ans = np.array([
            [0, 0, -1, 10, 0, 0],
            [0, 0, 111, -1.9, 0, 0],
        ])
        self.assertTrue(np.array_equal(ans, self.w.mcw))

        self.w.displace_mcw(-1)
        self.assertTrue(np.array_equal(original_mcw, self.w.mcw))

    def test_get_encoded_model(self):
        file = 'mcw2_4.txt'
        message_length = 4
        self.w.mcw_from_coefficients(file, message_length)
        encoded_output = self.w.get_encoded_output()
        self.assertTrue(np.array_equal(np.zeros(4, ), encoded_output))

    def test_encode(self):
        file = 'mcw2_4.txt'
        message = np.array([-1, 1, -1, 1])
        self.w.mcw_from_coefficients(file, np.size(message))

        ans = np.matmul(np.array([[-1, 10, 0, 0], [-111, -1.9, 0, 0], [0, 0, -1, 10], [0, 0, -111, -1.9]]), message)

        # model = self.w.get_encoded_output()
        # self.assertEqual(np.size(model), np.size(ans))

        encoded_output = self.w.encode(message)
        self.assertTrue(np.array_equal(ans, encoded_output))


class WaveletLongTests(unittest.TestCase):
    def setUp(self):
        self.w = Wavelet(m=2, g=64)

    @staticmethod
    def read_256_coefficients():
        with open('mcw2_128.txt', 'r') as file:
            flatten_array = np.asarray([float(l) for l in file.readlines()])
            return flatten_array[:128], flatten_array[128:]

    def test_get_raw_128_lines(self):
        file = 'mcw2_128.txt'
        ans0, ans1 = self.read_256_coefficients()
        l0, l1 = list(self.w._get_raw_lines(file))
        self.assertTrue(np.array_equal(ans0, l0))
        self.assertTrue(np.array_equal(ans1, l1))
        self.assertEqual(np.size(l0), self.w.mg)
        self.assertEqual(np.size(l1), self.w.mg)


if __name__ == '__main__':
    unittest.main()
