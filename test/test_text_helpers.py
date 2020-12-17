from collections import namedtuple
import unittest
from lstm_word_segmentation.text_helpers import remove_tags, clean_line, normalize_string


class TestRemoveTags(unittest.TestCase):
    def test_remove_tags(self):
        TestCase = namedtuple("TestCase", ["line", "st_tag", "fn_tag", "expected"])
        cases = [
            TestCase("Word segmentation is an interesting problem.", "<NE>", "</NE>",
                     "Word segmentation is an interesting problem."),
            TestCase("|Word| |segmentation| |is| |an| |interesting| |problem|.|", "<NE>", "</NE>",
                     "|Word| |segmentation| |is| |an| |interesting| |problem|.|"),
            TestCase("Word| |segmentation| |is| |an| |interesting| |problem|.|", "<NE>", "</NE>",
                     "Word| |segmentation| |is| |an| |interesting| |problem|.|"),
            TestCase("Word| |segmentation| |is| |an| |interesting| |problem|.", "<NE>", "</NE>",
                     "Word| |segmentation| |is| |an| |interesting| |problem|."),
            TestCase("|Word| |segmentation| |is| |an| |interesting| |problem|.|", "<NE>", "</NE>",
                     "|Word| |segmentation| |is| |an| |interesting| |problem|.|"),
            TestCase("|Word| |<NE>segmentation is</NE>| |an| |interesting| |problem|.|", "<NE>", "</NE>",
                     "|Word| | |an| |interesting| |problem|.|"),
            TestCase("|Word| |<NE>segmentation is</NE>|an| |interesting| |problem|.|", "<NE>", "</NE>",
                     "|Word| |an| |interesting| |problem|.|"),
            TestCase("|Word|<NE>segmentation is</NE>| |an| |interesting| |problem|.|", "<NE>", "</NE>",
                     "|Word| |an| |interesting| |problem|.|"),
            TestCase("|Word|<NE>segmentation is</NE>|an| |interesting| |problem|.|", "<NE>", "</NE>",
                     "|Word|an| |interesting| |problem|.|"),
            TestCase("<NE>segmentation is</NE>|an| |interesting| |problem|.|", "<NE>", "</NE>",
                     "|an| |interesting| |problem|.|"),
            TestCase("|<NE>segmentation is</NE>|an| |interesting| |problem|.|", "<NE>", "</NE>",
                     "|an| |interesting| |problem|.|"),
            TestCase("|word| |<NE>segmentation is</NE>|<NE>an interesting</NE>| |problem|.|", "<NE>", "</NE>",
                     "|word| | |problem|.|"),
            TestCase("|word| |<NE>segmentation is</NE>|    <NE>an interesting</NE>| |problem|.|", "<NE>", "</NE>",
                     "|word| | |problem|.|"),
            TestCase("|word| |<NE>segmentation is</NE>| <NE>an interesting</NE> blah | |problem|.|", "<NE>", "</NE>",
                     "|word| | |problem|.|"),
            TestCase("|word| |<NE>segmentation is</NE>| |<NE>an interesting</NE>| |problem|.|", "<NE>", "</NE>",
                     "|word| | | |problem|.|"),
            TestCase("|word|<NE>segmentation is</NE>|<NE>an interesting</NE>| |problem|.|", "<NE>", "</NE>",
                     "|word| |problem|.|"),
            TestCase("|word| |<NE>segmentation is</NE>|", "<NE>", "</NE>",
                     "|word| |"),
            TestCase("|word| |<NE>segmentation is</NE>", "<NE>", "</NE>",
                     "|word| |"),
            TestCase("|word| <NE>segmentation is</NE>", "<NE>", "</NE>",
                     "|word|"),
            TestCase("|word| <NE>segmentation is</NE>", "<NE>", "</NE>|",
                     "|word|"),
            TestCase("|<NE>segmentation is</NE>", "<NE>", "</NE>",
                     "|"),
            TestCase("|<NE>segmentation is</NE>|", "<NE>", "</NE>",
                     "|"),
            TestCase("<NE>segmentation is</NE>", "<NE>", "</NE>",
                     ""),
            TestCase("|<NE>segmentation is</NE>|", "<NE>", "</NE>",
                     "|"),
            TestCase("word|<NE>segmentation| |is</NE>|an|interesting|problem|", "<NE>", "</NE>",
                     "word|an|interesting|problem|"),
            TestCase("word|<NE>segmentation is</NE>|<AB>an interesting</AB>|problem|", "<NE>", "</NE>",
                     "word|<AB>an interesting</AB>|problem|"),
            TestCase("word|<NE>segmentation is</NE>|<AB>an interesting</AB>|problem| ", "<NE>", "</NE>",
                     "word|<AB>an interesting</AB>|problem| "),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะกะเ</AB>| ", "<NE>", "</NE>",
                     "|วลาเอ|<AB>แม้จะกะเ</AB>| "),
            TestCase("|วลาเอ|<AB>แม้จะกะเ</AB>| ", "<AB>", "</AB>",
                     "|วลาเอ| "),
        ]
        for cas in cases:
            computed = remove_tags(line=cas.line, st_tag=cas.st_tag, fn_tag=cas.fn_tag)
            self.assertEqual(cas.expected, computed)


class TestCleanLine(unittest.TestCase):
    def test_clean_line(self):
        TestCase = namedtuple("TestCase", ["line", "segmented", "expected"])
        cases = [
            TestCase("Word segmentation is https an interesting problem.", True, -1),
            TestCase("Word segmentation is an interesting problem.", True, -1),
            TestCase("|Word| |segmentation| |is| |an| |interesting| |problem|.|", True, -1),
            TestCase("|", True, -1),
            TestCase("", True, -1),
            TestCase(" ", True, -1),
            TestCase("| |", True, -1),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|", True, -1),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>| |<AB>แม้จะกะเวลาเอ/AB>", True, -1),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะกะเ</AB>", True, "|วลาเอ|"),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะกะเ</AB>|", True, "|วลาเอ|"),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะกะเ</AB>| |", True, "|วลาเอ| |"),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะ https กะเ</AB>| |", True, -1),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะกะเ</AB>|a", True, "|วลาเอ|a|"),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะกะเ</AB>|ab", True, "|วลาเอ|ab|"),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะกะเ</AB>|ab|", True, "|วลาเอ|ab|"),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะกะเ</AB>|ab |", True, "|วลาเอ|ab |"),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะกะเ</AB>| |", True, "|วลาเอ| |"),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะกะเ</AB>| ", True, "|วลาเอ|"),
            TestCase("<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะกะเ</AB>| ", True, "|วลาเอ|"),
            TestCase("word segmentation is interesting วลาเอ ", True, "|word segmentation is interesting วลาเอ|"),
            TestCase("  word segmentation is interesting วลาเอ ", True, "|word segmentation is interesting วลาเอ|"),
            TestCase("  |word segmentation is interesting วลาเอ ", True, "|word segmentation is interesting วลาเอ|"),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>| |<AB>แม้จะกะเวลาเอ/AB>", False, -1),
            TestCase("<NE>แม้จะกะเวลาเอา</NE>| |<AB>แม้จะกะเวลาเอ/AB>", False, -1),
            TestCase("word segmentation is interesting วลาเอ ", False, "word segmentation is interesting วลาเอ"),
            TestCase("|<NE>แม้จะกะเวลาเอา</NE>|วลาเอ|<AB>แม้จะกะเ</AB>| ", False, "|วลาเอ|"),
            TestCase("  |word segmentation is interesting วลาเอ ", False, "|word segmentation is interesting วลาเอ"),
            TestCase("   word segmentation is วลาเอ   |", False, "word segmentation is วลาเอ   |"),

        ]
        for cas in cases:
            computed = clean_line(line=cas.line, segmented=cas.segmented)
            self.assertEqual(cas.expected, computed)


class TestNormalizeString(unittest.TestCase):
    def test_normalize_string(self):
        TestCase = namedtuple("TestCase", ["in_str", "scripts", "out_str"])
        cases = [
            TestCase("abc", ["Latn"], "abc"),
            TestCase("abc", [], "LLL"),
            TestCase("a𑄌", ["Latn", "Cakm"], "a𑄌"),
            TestCase("a𑄌", ["Latn"], "a𑄃"),
            TestCase("a𑄌", ["Cakm"], "L𑄌"),
            TestCase("a𑄌", [], "L𑄃"),
            # NOTE: ASCII digits have script Common, not Latin
            TestCase("123", ["Latn"], "000"),
            TestCase("၁၁၁", ["Mymr"], "၁၁၁"),
            TestCase("၁၁၁", [], "၀၀၀"),
            # NOTE: Currency symbols have script Common
            TestCase("฿100", ["Thai"], "$000"),
            TestCase("฿100", [], "$000"),
            TestCase(
                "พันโทหญิง สมเด็จพระนางเจ้าอินทรศักดิศจี พระวรราชชายา (พระนามเดิม: ประไพ; 10 มิถุนายน พ.ศ. 2445 — 30 พฤศจิกายน พ.ศ. 2518) พระวรราชชายาในพระบาทสมเด็จพระมงกุฎเกล้าเจ้าอยู่หัว",
                ["Thai"],
                "พันโทหญิง สมเด็จพระนางเจ้าอินทรศักดิศจี พระวรราชชายา (พระนามเดิม: ประไพ; 00 มิถุนายน พ.ศ. 0000 — 00 พฤศจิกายน พ.ศ. 0000) พระวรราชชายาในพระบาทสมเด็จพระมงกุฎเกล้าเจ้าอยู่หัว"),
            TestCase(
                "พันโทหญิง สมเด็จพระนางเจ้าอินทรศักดิศจี พระวรราชชายา (พระนามเดิม: ประไพ; 10 มิถุนายน พ.ศ. 2445 — 30 พฤศจิกายน พ.ศ. 2518) พระวรราชชายาในพระบาทสมเด็จพระมงกุฎเกล้าเจ้าอยู่หัว",
                [],
                "ทัทททททิท ทททท็ททททททททท้ททิททททัททิทที ทททททททททททท (ททททททททิท: ททททท; 00 ทิทุทททท ท.ท. 0000 — 00 ททททิทททท ท.ท. 0000) ทททททททททททททททททททททททท็ทททททททุทททท้ททท้ทททู่ทัท"),
        ]
        for cas in cases:
            actual = normalize_string(in_str=cas.in_str, allowed_scripts=cas.scripts)
            self.assertEqual(cas.out_str, actual, cas)


if __name__ == "__main__":
    unittest.main()
