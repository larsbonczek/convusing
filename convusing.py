from abc import ABC
import numpy as np
from numpy.lib.function_base import copy
from scipy.signal import convolve2d
from typing import List
import re
import sys


class Filter(ABC):

    def apply(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class ConvFilter(Filter):

    def __init__(self, weights: np.ndarray, bias: float = 0) -> None:
        super().__init__()
        assert len(weights.shape) == 2
        assert weights.shape[0] == weights.shape[1]
        self.weights = weights
        self.bias = bias

    def apply(self, data: np.ndarray) -> np.ndarray: 
        return convolve2d(data, self.weights[::-1,::-1], mode='same') + self.bias


class Activation(ABC):

    def apply(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class ReLUActivation(Activation):

    def __init__(self, threshold: float = 0) -> None:
        super().__init__()
        self.threshold = threshold

    def apply(self, data: np.ndarray) -> np.ndarray:
        data = data.copy()
        data[data < self.threshold] = self.threshold
        return data


class ConvUsingProgram:

    def __init__(self, filters: List[Filter], activation: Activation) -> None:
        self.filters = filters
        print('filters:')
        print('\n'.join([str(filter.weights) for filter in filters]))
        self.activation = activation

    def run(self, data: np.ndarray) -> np.ndarray:
        states = set()
        print('run:')
        while True:
            print(data)
            #input('step?')

            result = self.step(data)

            res_key = result.tobytes()
            if res_key in states:
                return result
            states.add(res_key)

            data = result

    def step(self, data: np.ndarray) -> np.ndarray:
        filter_result = np.array([filter.apply(data) for filter in self.filters])
        #print(filter_result)
        activation_result = self.activation.apply(filter_result)
        #print(activation_result)
        return np.sum(activation_result, axis=0)


def parse_program(code: str) -> ConvUsingProgram:
    code_lines = _parse_lines(code)
    filters = []
    i = 0
    while i < len(code_lines):
        size = len(code_lines[i])
        if size == 2:
            size = 1
        assert size % 2 != 0

        weights = np.nan_to_num(np.array(code_lines[i:i+size]), copy=False)
        bias = code_lines[i+size][0]

        filters.append(ConvFilter(weights, bias=bias))
        i += size + 1

    activation = ReLUActivation(threshold=0)
    return ConvUsingProgram(filters, activation)


def parse_data(data: str) -> np.ndarray:
    data_lines = _parse_lines(data)
    return np.nan_to_num(np.array(data_lines), copy=False)


def _parse_lines(text: str) -> List[List[float]]:
    return [
        code_line
        for code_line in [
            [float(number) for number in re.findall('([+-]?(?:\d+\.?\d*|\.\d+|inf))', line)] 
            for line in text.splitlines()
        ]
        if len(code_line) > 0
    ]


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        program = parse_program(f.read())

    with open(sys.argv[2]) as f:
        data = parse_data(f.read())

    result = program.run(data)

    print('result:')
    print(result)