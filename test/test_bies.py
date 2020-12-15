from collections import namedtuple
import unittest
from lstm_word_segmentation.bies import Bies


class TestBies(unittest.TestCase):
    def test_normalize_bies(self):
        TestCase = namedtuple("TestCase", ["str", "expected"])
        cases = [
            TestCase("biiie", "biiie"),
            TestCase("biiiebie", "biiiebie"),
            TestCase("bebiie", "bebiie"),
            TestCase("bies", "bies"),
            TestCase("ssbesbie", "ssbesbie"),
            TestCase("ssssss", "ssssss"),
            TestCase("bebebebes", "bebebebes"),
            TestCase("e", "s"),
            TestCase("b", "s"),
            TestCase("i", "s"),
            TestCase("s", "s"),
            TestCase("bi", "be"),
            TestCase("ii", "be"),
            TestCase("ie", "be"),
            TestCase("is", "ss"),
            TestCase("biiis", "biies"),
            TestCase("bbiiis", "biiies"),
            TestCase("sbiiis", "sbiies"),
            TestCase("biisiiie", "biesbiie"),
            TestCase("biisiiie", "biesbiie"),
            TestCase("ibiie", "sbiie"),
            TestCase("iibiie", "bebiie"),
            TestCase("iiebiie", "biebiie"),
            TestCase("isbiie", "ssbiie"),
            TestCase("eebiie", "bebiie"),
            TestCase("eibiie", "bebiie"),
            TestCase("esbiie", "ssbiie"),
            TestCase("biieesbiie", "biiessbiie"),
        ]
        for cas in cases:
            bies = Bies(input_bies=cas.str, input_type="str")
            bies.normalize_bies()
            self.assertEqual(cas.expected, bies.str)

if __name__ == "__main__":
    unittest.main()
