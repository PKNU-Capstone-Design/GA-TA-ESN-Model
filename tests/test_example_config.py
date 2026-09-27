import unittest

from example import FULL_EXPERIMENT, QUICK_CHECK, get_experiment_config


class ExperimentConfigTest(unittest.TestCase):
    def test_full_experiment_is_the_default(self):
        config = get_experiment_config()

        self.assertEqual(FULL_EXPERIMENT, config)
        self.assertEqual(5, config.n_splits)
        self.assertEqual(30, config.pop_size)
        self.assertEqual(30, config.num_generations)

    def test_quick_check_uses_reduced_values(self):
        config = get_experiment_config(quick=True)

        self.assertEqual(QUICK_CHECK, config)
        self.assertEqual(2, config.n_splits)
        self.assertEqual(4, config.pop_size)
        self.assertEqual(2, config.num_generations)


if __name__ == "__main__":
    unittest.main()
