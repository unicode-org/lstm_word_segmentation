from collections import namedtuple
import unittest
from lstm_word_segmentation.helpers import is_ascii, diff_strings, sigmoid


class TestIsAscii(unittest.TestCase):
    def test_is_ascii(self):
        TestCase = namedtuple("TestCase", ["str", "expected"])
        cases = [
            TestCase("Unicode", True),
            TestCase("123", True),
            TestCase("(hello!)", True),
            TestCase("\n", True),
            TestCase("", True),
            TestCase(" ", True),
            TestCase("กระสับกระส่ายและปวดศีรษะ", False),
            TestCase("กระสับ 123", False),
            TestCase("กร|ะสับ|กระส่ายและ|ปวดศี|รษะ", False),
            TestCase("Unicodeบ", False),
        ]
        for cas in cases:
            computed = is_ascii(input_str=cas.str)
            self.assertEqual(cas.expected, computed)


class TestDiffStrings(unittest.TestCase):
    def test_diff_string(self):
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
            computed = diff_strings(str1=cas.str1, str2=cas.str2)
            self.assertEqual(cas.expected, computed)


class TestSigmoid(unittest.TestCase):
    def test_sigmoid(self):
        TestCase = namedtuple("TestCase", ["x", "expected"])
        cases = [
            TestCase(0, 0.5),
            TestCase(1, 0.73105858),
            TestCase(2, 0.88079708),
            TestCase(3, 0.95257413),
            TestCase(10, 0.99995460),
            TestCase(100, 1),
            TestCase(1000, 1),
            TestCase(-1, 0.26894142),
            TestCase(-2, 0.11920292),
            TestCase(-3, 0.04742587),
            TestCase(-10, 0.00004540),
            TestCase(-100, 0),
            TestCase(-1000, 0),
        ]
        for cas in cases:
            computed = sigmoid(x=cas.x)
            self.assertAlmostEqual(cas.expected, computed)


if __name__ == "__main__":
    unittest.main()
