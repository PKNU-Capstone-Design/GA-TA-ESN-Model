import unittest

import numpy as np
import pandas as pd

from MovingAverage import generate_MA_signals_vectorized


class MovingAverageSignalTest(unittest.TestCase):
    def test_signal_labels_accept_string_values_on_recent_pandas(self):
        index = pd.date_range("2020-01-01", periods=220)
        close = 100 + np.sin(np.arange(220) / 5) * 10
        data = pd.DataFrame({"Close": close}, index=index)

        result = generate_MA_signals_vectorized(
            data,
            41,
            5,
            2.0,
            0.5,
            0.2,
            2.0,
            0.5,
            0.2,
        )

        self.assertEqual(["Index", "Close", "Type"], result.columns.tolist())


if __name__ == "__main__":
    unittest.main()
