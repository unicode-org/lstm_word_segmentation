from .constants import CHAR_TYPE_TO_BUCKET
from icu import Char, Script, UCharCategory


def normalize_string(in_str, allowed_scripts):
    """
    Normalizes in_str by replacing letters and digits in other scripts with
    exemplar values.

    Args:
        in_str: String to process
        allowed_scripts: List of script short names (like "Mymr") to preserve
    """
    # TODO: Consider checking ScriptExtensions here as well
    output = ""
    for ch in in_str:
        ch_script = Script.getScript(ch)
        ch_type = Char.charType(ch)
        ch_bucket = CHAR_TYPE_TO_BUCKET[ch_type]
        ch_digit = Char.digit(ch)
        if ch_script.getShortName() in allowed_scripts:
            # ch is in an allowed script:
            # copy directly to the output
            output += ch
        elif ch_bucket == 1:
            # ch is a letter in a disallowed script:
            # normalize to the sample char for that script
            output += Script.getSampleString(ch_script)
        elif ch_bucket == 3 and ch_digit != -1:
            # ch is a decimal digit in a disallowed script:
            # normalize to the zero digit in that numbering system
            output += chr(ord(ch) - ch_digit)
        elif ch_type == UCharCategory.CURRENCY_SYMBOL:
            # ch is a currency symbol in a disallowed script:
            # normalize to $
            output += "$"
        else:
            # all other characters:
            # copy directly to the output
            output += ch
    return output
