#!/usr/bin/env python3

import sys
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

class FileParser:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.header = None
        self.data = None
        self._parse()

    def _parse(self):
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        # First line is the header (number of data points)
        self.header = int(lines[0].strip())

        # Parse the data lines
        data_lines = []
        for line in lines[1:]:
            assert line.strip()
            values = [float(x) for x in line.strip().split()]
            assert len(values) == 4
            data_lines.append(values)

        self.data = np.array(data_lines)

def compare_with(p1, p2):
    assert p1.header == p2.header
    assert p1.data.shape == p2.data.shape
    p1_c1, p1_c2, p1_c3, p1_c4 = p1.data.T
    p2_c1, p2_c2, p2_c3, p2_c4 = p2.data.T
    assert np.allclose(p1_c1, p2_c1, atol=0.1)
    assert np.allclose(p1_c2, p2_c2, atol=0.1)
    assert np.allclose(p1_c3, p2_c3, atol=0.5)

if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    parser1 = FileParser(file1)
    parser2 = FileParser(file2)
    compare_with(parser1, parser2)
