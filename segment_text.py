# Copyright (C) 2021 and later: Unicode, Inc. and others.
# License & terms of use: http://www.unicode.org/copyright.html
# Lint as: python3
from lstm_word_segmentation.word_segmenter import pick_lstm_model
import glob, sys, getopt

"""
A sample / simple program to segment the text from standard input
and output the input with segmented result to standout output.
"""

model_name = "Thai_codepoints_exclusive_model4_heavy"

def available_models():
  return [ m.replace("Models/", "") \
          .replace("/weights.json", "") \
          for m in glob.glob("Models/*/weights.json")]

def print_models():
  print("Supported Models")
  for m in sorted(available_models()):
    if m == model_name:
      print("  ", m, "[DEFAULT]")
    else:
      print("  ", m)

def print_usage():
  print('segment_text.py -h -l -m model')
  print("""
        -h      \tHelp / Usage
        -l      \tList models
        -m model\tSpecify model
        """)
  print_models()

def embedding_from_name(name):
  if "_codepoints_" in name:
    return "codepoints"
  else:
    return "grapheme_clusters_tf"

def main(argv):
   global model_name
   try:
     opts, args = getopt.getopt(argv,"hlm::")
   except getopt.GetoptError:
     print_usage()
     sys.exit(2)
   for opt, arg in opts:
      if opt == '-m':
        model_name = arg
      if opt == '-h':
        print_usage()
        sys.exit()
      if opt == '-l':
        print_models()
        sys.exit()

   file1 = sys.stdin
   Lines = file1.readlines()

   embedding = embedding_from_name(model_name)
   word_segmenter = pick_lstm_model(model_name=model_name, embedding=embedding,
                                    train_data="", eval_data="")

   print("Model:", model_name)
   print("Embedding:", embedding)

   count = 0
   # Strips the newline character
   for line in Lines:
     line = line.strip()
     print("Input:\t", line)
     print("Output:\t", word_segmenter.segment_arbitrary_line(line))

if __name__ == "__main__":
  main(sys.argv[1:])
