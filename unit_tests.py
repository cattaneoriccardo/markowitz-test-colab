import unittest
import assets as at

class TestAssets(unittest.TestCase):
    def test_add_or_modify_asset(self):
        """
        Test that it can sum a list of integers
        """
        assets = at.Assets()
        result = sum(assets)
        self.assertEqual(result, 6)

    def test_list_fraction(self):
        """
        Test that it can sum a list of fractions
        """
        data = [Fraction(1, 4), Fraction(1, 4), Fraction(2, 5)]
        result = sum(data)
        self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()
