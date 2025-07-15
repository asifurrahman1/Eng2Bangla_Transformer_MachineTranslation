import re
from indicnlp.tokenize import indic_tokenize

class BaseTokenizer:
    def src_tokenizer(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def target_tokenizer(self, text):
        return [t for t in indic_tokenize.trivial_tokenize(text, lang='bn') if t not in ['।', '.', ',', '?', '!', '(', ')', ':', ';', '"']]
      
      