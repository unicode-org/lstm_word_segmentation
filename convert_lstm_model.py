# Copyright (C) 2021 and later: Unicode, Inc. and others.
# License & terms of use: http://www.unicode.org/copyright.html
# Lint as: python3

import sys, getopt, json, struct

"""
Tool to convert Models/*/weights.json files into a resource file could be build
into ICU. The result should be copy to icu/icu4c/source/data/brkitr/lstm.
See https://docs.google.com/document/d/1EVK2CwOmUamJwMOMbbdTz7tuaV0IR21rMoH7a3pyFwE/edit#heading=h.qkedw6o6vy20
for detail design.
"""

def main(argv):
   inputfile = ''
   script = 'Thai'
   type = "codepoint"
   try:
     opts, args = getopt.getopt(argv,"hist::",["ifile=","script=","type="])
   except getopt.GetoptError:
     print('convert_lstm_model.py -i <inputfile> -s <script> -t (*codepoint*|graphemecluster)')
     sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
        print('convert_lstm_model.py -i <inputfile> -s <script> -t (*codepoint*|graphemecluster)')
        sys.exit()
      elif opt in ("-i", "--ifile"):
        inputfile = arg
      elif opt in ("-s", "--script"):
        script = arg
      elif opt in ("-t", "--type"):
        type = arg

   input = json.load(open(inputfile, 'r'))
   embeddings = input["mat1"]["dim"][1];
   hunits = input["mat3"]["dim"][0];
   dict_size = len(input["dic"])
   model = input["model"]

   verify_dimension(input, dict_size, embeddings, hunits)

   copyright="""// © 2021 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html"""
   print(copyright)
   print("{script}:table(nofallback){{".format(script=script))
   print("    model{{\"{model}\"}}".format(model=model))
   print("    type{{\"{type}\"}}".format(type=type))
   print("    embeddings:int{{{embeddings}}}".format(embeddings=embeddings))
   print("    hunits:int{{{hunits}}}".format(hunits=hunits))
   print_dict(input["dic"])
   print("    data:intvector{")
   print_flaot_in_int(input["mat1"]["data"])
   print_flaot_in_int(input["mat2"]["data"])
   print_flaot_in_int(input["mat3"]["data"])
   print_flaot_in_int(input["mat4"]["data"])
   print_flaot_in_int(input["mat5"]["data"])
   print_flaot_in_int(input["mat6"]["data"])
   print_flaot_in_int(input["mat7"]["data"])
   print_flaot_in_int(input["mat8"]["data"])
   print_flaot_in_int(input["mat9"]["data"])
   print("    }")
   print("}")

def print_dict(dict):
   print("    dict{")
   i = 0
   for k in dict:
     print("        \"{key}\",".format(key=k))
     if i != dict[k]:
       print("Incorrect value for dic \"{k}\": {v}- expecting {i}"
             .format(k=k, v=dict[k], i=i))
       sys.exit(2)
     i += 1
   print("    }")

def print_flaot_in_int(data):
   # TODO currently we print each float as 32 bit int. We may later change it to
   # print two float as float16 into one 32 bit int.
   for i in data:
     print("        {f},".format(f=struct.unpack("i", struct.pack("f", i))[0]))

def verify_dimension(input, dict_size, embeddings, hunits):
   hunits4 = 4 * hunits
   hunits2 = 2 * hunits
   if (input["mat1"]["dim"][0] != dict_size + 1):
     dimension_error("mat1", dict_size + 1, input["mat1"]["dim"])
   if (input["mat1"]["dim"][1] != embeddings):
     dimension_error("mat1", embeddings, input["mat1"]["dim"])
   if (input["mat2"]["dim"][0] != embeddings):
     dimension_error("mat2", embeddings, input["mat2"]["dim"])
   if (input["mat2"]["dim"][1] != hunits4):
     dimension_error("mat2", hunits4, input["mat2"]["dim"])
   if (input["mat3"]["dim"][0] != hunits):
     dimension_error("mat3", hunits, input["mat3"]["dim"])
   if (input["mat3"]["dim"][1] != hunits4):
     dimension_error("mat3", hunits4, input["mat3"]["dim"])
   if (input["mat4"]["dim"][0] != hunits4):
     dimension_error("mat4", hunits4, input["mat4"]["dim"])
   if (input["mat5"]["dim"][0] != embeddings):
     dimension_error("mat5", embeddings, input["mat5"]["dim"])
   if (input["mat5"]["dim"][1] != hunits4):
     dimension_error("mat5", hunits4, input["mat5"]["dim"])
   if (input["mat6"]["dim"][0] != hunits):
     dimension_error("mat6", hunits, input["mat6"]["dim"])
   if (input["mat6"]["dim"][1] != hunits4):
     dimension_error("mat6", hunits4, input["mat6"]["dim"])
   if (input["mat7"]["dim"][0] != hunits4):
     dimension_error("mat7", hunits4, input["mat7"]["dim"])
   if (input["mat8"]["dim"][0] != hunits2):
     dimension_error("mat8", hunits2, input["mat8"]["dim"])
   if (input["mat8"]["dim"][1] != 4):
     dimension_error("mat8", 4, input["mat8"]["dim"])
   if (input["mat9"]["dim"][0] != 4):
     dimension_error("mat9", 4, input["mat9"]["dim"])

def dimension_error(name, value, dim):
   print("Dimension mismatch for {name}, expected {value}, but got {dim}"
         .format(name=name, value=value, dim=dim))
   sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1:])
