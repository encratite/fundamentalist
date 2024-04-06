import math

class NormalizationTracker:
    def __init__(self, group_limit = 10000):
        self._index = 0
        self._max_values = []
        self._group_limit = group_limit

    def handle_value(self, value):
        abs_value = math.fabs(value)
        if self._index < len(self._max_values):
            if self._max_values[self._index] < abs_value:
                self._max_values[self._index] = abs_value
        else:
            self._max_values.append(abs_value)
        self._index += 1

    def reset(self):
        self._index = 0

    def normalize(self, tensors):
        group_max = None
        for max_value in self._max_values:
            if group_max is None or max_value > group_max:
                group_max = max_value
        for tensor in tensors:
            for i in range(len(tensor)):
                max_value = self._max_values[i]
                divisor = max_value
                if max_value >= self._group_limit:
                    divisor = max_value
                if divisor > 0:
                    tensor[i] /= divisor