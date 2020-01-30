#!/bin/python3

import numpy as np


class Wavelet:
    def __init__(self, m, g):
        self.m = m
        self.g = g
        self.mg = m * g
        self.A = None
        self.mcw = None

    def _allocate_a_matrix(self):
        if self.A is None:
            self.A = np.zeros((self.m, self.mg))

    def _get_raw_lines(self, file, type=float):
        with open(file, 'r') as file:
            flatten = [type(l) for l in file.readlines()]
            size = int(np.size(flatten))
            step = int(size / self.m)
            for i in np.arange(0, size, step):
                yield flatten[i: i + step]

    def _set_a_coefficients(self, file):
        """Loads A with coefficients defined on raw file `file`"""
        self._allocate_a_matrix()
        lines = self._get_raw_lines(file)
        for i in range(self.m):
            self.A[i] = next(lines)

    def _allocate_mcw(self, message_length):
        """MCW matrix will be rotated, so it has m as number of lines"""
        if self.mcw is None:
            self.mcw = np.zeros((self.m, message_length + self.mg - self.m))

    def check_mcw_existence(self):
        if self.mcw is None:
            raise ValueError("MCW is not known")

    def _get_encoded_model(self):
        self.check_mcw_existence()
        return np.zeros((self.mcw.shape[1],))

    def mcw_from_coefficients(self, file, message_length):
        self._set_a_coefficients(file)
        self._allocate_mcw(message_length)
        self.mcw[:self.m, :self.mg] = self.A

    def encode(self, message):
        encoded = self._get_encoded_model()
        mcw = np.copy(self.mcw)
        for i in range(0, np.size(message), self.m):
            encoded += np.matmul(message[i: i + self.m], mcw)
            mcw = np.roll(mcw, self.m)
        return encoded

    def _get_decoded_model(self):
        self.check_mcw_existence()
        return np.zeros((self.mcw.shape[1] - self.mg + self.m,))

    def decode(self, encoded):
        decoded = self._get_decoded_model()
        mcw = np.copy(self.mcw)
        for i in range(np.size(decoded)):
            if i and i % self.m == 0:
                mcw = np.roll(mcw, self.m)
            decoded[i] = np.dot(mcw[i % self.m], encoded)
        return decoded
