import unittest

import numpy as np

from wavelet import Wavelet


class OtherTests(unittest.TestCase):
    def test_vstack(self):
        arr1 = np.array([0, 1])
        arr2 = np.array([2, 3])
        ret = np.vstack((arr1, arr2))
        ans = np.array([[0, 1], [2, 3]])
        self.assertTrue(np.array_equal(ret, ans))

    def test_elementwise_multiplication(self):
        arr1 = np.array([10, -10, 1, -1, 1, -1])
        arr2 = np.array([[1, 1, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0]])
        ans = np.array([[10, -10, 0, 0, 0, 0], [10, 10, 0, 0, 0, 0]])
        self.assertTrue(np.array_equal(ans, arr1 * arr2))


class WaveletShortTests(unittest.TestCase):
    def setUp(self):
        self.w = Wavelet(m=2, g=1)
        self.file = 'mcw2_4.txt'
        self.message = np.array([-1, 1, -1, 1])

    def test_mg(self):
        self.assertEqual(self.w.mg, 2)

    def test_get_raw_4_lines(self):
        l0, l1 = list(self.w._get_raw_lines(self.file))
        self.assertTrue(np.array_equal(l0, np.array([1, 1])))
        self.assertTrue(np.array_equal(l1, np.array([1, -1])))

    def test_allocate_matrix(self):
        self.assertEqual(self.w.A, None)
        ans = np.zeros((self.w.m, self.w.mg))
        self.w._allocate_a_matrix()
        self.assertTrue(np.array_equal(ans, self.w.A))

    def test_set_a_coefficients(self):
        self.w = Wavelet(m=2, g=1)
        ans = np.array([[1, 1], [1, -1]])
        self.w._set_a_coefficients(self.file)
        self.assertTrue(np.array_equal(ans, self.w.A))

    def test_allocate_mcw(self):
        message_length = 4
        self.w._allocate_mcw(message_length)
        ans = np.zeros((self.w.m, message_length + self.w.mg - self.w.m))
        self.assertTrue(np.array_equal(self.w.mcw, ans))

    def test_mcw_from_coefficients(self):
        message_length = 4
        self.w.mcw_from_coefficients(self.file, message_length)
        mcw_lines, mcw_columns = self.w.mcw.shape
        self.assertEqual(self.w.m, mcw_lines)
        self.assertEqual(message_length + self.w.mg - self.w.m, mcw_columns)
        self.assertTrue(np.array_equal(self.w.mcw[:self.w.m, :self.w.mg], self.w.A))

    def test_get_encoded_model(self):
        message_length = 4
        self.w.mcw_from_coefficients(self.file, message_length)
        encoded_output = self.w._get_encoded_model()
        self.assertTrue(np.array_equal(np.zeros(message_length, ), encoded_output))

    def test_encoding(self):
        message = np.array([-1, -1, -1, -1, -1, -1])
        self.w.mcw_from_coefficients(self.file, np.size(message))

        ans = np.matmul(message, np.array([
            [1, 1, 0, 0, 0, 0],
            [1, -1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, -1]
        ]))

        encoded_output = self.w.encode(message)
        self.assertTrue(np.array_equal(ans, encoded_output))

    def test_get_decoded_model(self):
        self.assertRaises(ValueError, self.w._get_decoded_model)
        self.w.mcw_from_coefficients(self.file, np.size(self.message))
        decoded_model = self.w._get_decoded_model()
        self.assertEqual(np.size(self.message), np.size(decoded_model))

    def test_decoding(self):
        self.w.mcw_from_coefficients(self.file, np.size(self.message))
        encoded = self.w.encode(self.message)
        decoded = self.w.decode(encoded)
        decoded[decoded >= 0] = 1
        decoded[decoded < 0] = -1
        self.assertTrue(np.array_equal(self.message, decoded))


class WaveletLongTests(unittest.TestCase):
    def setUp(self):
        self.w = Wavelet(m=2, g=64)
        self.file = 'mcw2_128.txt'
        self.message = np.array([1, -1, 1, 1, 1, -1])

    def read_256_coefficients(self):
        with open(self.file, 'r') as file:
            flatten_array = np.asarray([float(l) for l in file.readlines()])
            return flatten_array[:128], flatten_array[128:]

    def test_get_raw_128_lines(self):
        ans0, ans1 = self.read_256_coefficients()
        l0, l1 = list(self.w._get_raw_lines(self.file))
        self.assertTrue(np.array_equal(ans0, l0))
        self.assertTrue(np.array_equal(ans1, l1))
        self.assertEqual(np.size(l0), self.w.mg)
        self.assertEqual(np.size(l1), self.w.mg)

    def test_encoding(self):
        n = np.size(self.message)
        mcw_lines = n
        mcw_columns = n + self.w.mg - self.w.m
        mcw = np.zeros((mcw_lines, mcw_columns))
        f_line, s_line = self.read_256_coefficients()
        for i in range(0, mcw_lines, self.w.m):
            mcw[i, i: i + self.w.mg] = f_line
            mcw[i + 1, i:i + self.w.mg] = s_line
        self.assertEqual(mcw.shape[0], 6)
        ans = np.matmul(self.message, mcw)

        self.w.mcw_from_coefficients(self.file, np.size(self.message))
        encoded = self.w.encode(self.message)

        self.assertTrue(np.array_equal(ans, encoded))

    def test_decoding(self):
        self.w.mcw_from_coefficients(self.file, np.size(self.message))
        encoded = self.w.encode(self.message)
        decoded = self.w.decode(encoded)
        decoded[decoded >= 0] = 1
        decoded[decoded < 0] = -1
        self.assertTrue(np.array_equal(self.message, decoded))

    def test_huge_encoding_and_decoding(self):
        message = np.array([np.random.choice([-1, 1]) for _ in range(50000)])
        self.w.mcw_from_coefficients(self.file, np.size(message))
        encoded = self.w.encode(message)
        encoded_elements = set(encoded)
        ans_elements = [-self.w.mg + k for k in range(2, 2 * self.w.mg + 1)]
        for el in encoded_elements:
            self.assertTrue(el in ans_elements)
        decoded = self.w.decode(encoded)
        self.assertEqual(len(set(np.abs(decoded))), 1)
        self.assertEqual(set(np.abs(decoded)), {self.w.mg})
        decoded[decoded >= 0] = 1
        decoded[decoded < 0] = -1
        self.assertTrue(np.array_equal(message, decoded))


if __name__ == '__main__':
    unittest.main()
