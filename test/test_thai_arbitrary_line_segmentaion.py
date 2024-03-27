import unittest
from lstm_word_segmentation.word_segmenter import pick_lstm_model


word_segmenter = pick_lstm_model(model_name="Thai_graphclust_model4_heavy", embedding="grapheme_clusters_tf",
                                 train_data="BEST", eval_data="BEST")



def transform_segmented_text(segmented_text):
    # Remove leading and trailing '|' characters, then split by '|'
    segmented_text = segmented_text.strip('|').split('|')
    # Filter out empty strings
    segmented_text = [segment for segment in segmented_text if segment]
    return segmented_text


class TestThaiSegmentation(unittest.TestCase):

    def test_complex_words(self):
        # Test cases for words or phrases
        self.assertEqual(transform_segmented_text(word_segmenter.segment_arbitrary_line("ประเทศไทย")), ["ประเทศ", "ไทย"])
        self.assertEqual(transform_segmented_text(word_segmenter.segment_arbitrary_line("กำลังเดินทาง")), ["กำลัง", "เดินทาง"])
        self.assertEqual(transform_segmented_text(word_segmenter.segment_arbitrary_line("หวังว่า")), ["หวัง", "ว่า"])

    
    # def test_punctuation(self):                                MAYBE??  not sure if thai uses punctuation marks
    #     # Test cases involving punctuation marks
    #     self.assertEqual(transform_segmented_text(word_segmenter.segment_arbitrary_line("ทำอะไร?")), ["ทำ", "อะไร", "?"])
    #     self.assertEqual(transform_segmented_text(word_segmenter.segment_arbitrary_line("ขอโทษครับ.")), ["ขอโทษ", "ครับ", "."])

    def test_numerals(self):
        # Test cases involving numerals
        self.assertEqual(transform_segmented_text(word_segmenter.segment_arbitrary_line("เดือน 12")), ['เดือน', ' ', '1', '2'])
        self.assertEqual(transform_segmented_text(word_segmenter.segment_arbitrary_line("ตั้งแต่ปี2020")), ["ตั้งแต่", "ปี", "2020"])

    
    # def test_mixed_text(self):                                MAYBE??
    #     # Test cases involving a mix of Thai and English words
    #     self.assertEqual(transform_segmented_text(word_segmenter.segment_arbitrary_line("สวัสดี Hello")), ["สวัสดี", "Hello"])
    #     self.assertEqual(transform_segmented_text(word_segmenter.segment_arbitrary_line("กินข้าว Rice")), ["กินข้าว", "Rice"])

    def test_special_characters(self):
        # Test cases involving special characters
        self.assertEqual(transform_segmented_text(word_segmenter.segment_arbitrary_line("@ฉันรักคุณ")), ["@", "ฉันรัก", "คุณ"])
        self.assertEqual(transform_segmented_text(word_segmenter.segment_arbitrary_line("ฉันรักคุณ#")), ["ฉันรัก", "คุณ", "#"])

if __name__ == '__main__':
    unittest.main()
