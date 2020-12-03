from collections import namedtuple
import unittest
from lstm_word_segmentation.helpers import diff_strings


class TestDiffStrings(unittest.TestCase):
    def test_normalize_string(self):
        TestCase = namedtuple("TestCase", ["str1", "str2", "expected"])
        cases = [
            TestCase("Google", "Google", 0),
            TestCase("Google", "Googee", 1),
            TestCase("Google", "google", 1),
            TestCase("Gooogl", "Google", 3),
            TestCase("Googl", "Google", -1),
            TestCase("", "Google", -1),
            TestCase("Google", "word segmentation", -1),
            TestCase("word segmentation", "werd segmentation", 1),
            TestCase("word segmentation", "wordsegmentation ", 13),
            TestCase("word segmentation", "wordsegmentation", -1),
            TestCase("sooo long", " sooolong", 3),

        ]
        for cas in cases:
            computed = diff_strings(cas.str1, cas.str2)
            self.assertEqual(cas.expected, computed)


if __name__ == "__main__":
    unittest.main()


