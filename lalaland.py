import regex
import pandas as pd
from collections import defaultdict


class BaseReplacer:

    def __init__(self, pattern_replace_pair_list=[]):
        self.pattern_replace_pair_list = pattern_replace_pair_list

    def transform(self, text):

        for pattern, replace in self.pattern_replace_pair_list:
            try:
                text = regex.sub(pattern, replace, text)
            except:
                pass
        return text.strip()


class LowerCaseConverter(BaseReplacer):

    def transform(self, text):
        return text.lower()


class TrashWords(BaseReplacer):

    def __init__(self):

        trash_words_reg = ['国产','授权','行货','原装','海淘','新西兰',
                           '法国','香港','港版','澳洲','英国',
                           '美国','美版','加拿大','意大利','德国',
                           '(荷兰(?!朵))','日本','韩国']

        self.pattern_replace_pair_list = [('|'.join(trash_words_reg), r'')]


class WordReplacer(BaseReplacer):

    def __init__(self):

        self.pattern_replace_pair_list = [(r'の', r'之')]


class BrandSplit(BaseReplacer):

    def transform(self, text):
        chn_brand = regex.findall(r'[\u4e00-\u9fff]+', text)
        en_brand = regex.findall(r'[a-z]+', text)
        return ''.join(chn_brand), ''.join(en_brand)


"""
class SymbolsCleaner(BaseReplacer):

    def __init__(self):
        self.pattern_replace_pair_list = \
            [r', ·＇+＋‘’!@#%&＆*?=_－—<>.··．﹒．:：;′()（）【】，。！\'？, /\\“”／の●', r""]
"""


class BrandSeparator:

    def __init__(self, brands):

        self.count_list = brands.value_counts()
        self.brands = list(self.count_list.index)

        self.new_brands = self.clean_brand()
        self.frame = self.get_frame()
        self.chn_name_list, self.en_name_list = self.merge_brand()

    def clean_brand(self):
        processors = [TrashWords(), WordReplacer(), LowerCaseConverter(), BrandSplit()]
        for processor in processors:
            self.brands = list(map(processor.transform, self.brands))
        return self.brands

    def get_frame(self):
        mapping_frame = pd.DataFrame(self.new_brands, index=list(self.count_list.index))
        mapping_frame['counts'] = self.count_list
        return mapping_frame

    def merge_brand(self):

        mapping_array = self.frame[[0, 1]].values

        chn_name_dict = defaultdict(list)
        en_name_dict = defaultdict(list)

        for line in mapping_array:
            if len(line[0]) > 1 and len(line[1]) > 1:
                chn_name_dict[line[0]].append(line[1])
                en_name_dict[line[1]].append(line[0])

        chn_name_list = chn_name_dict.items()
        en_name_list = en_name_dict.items()

        return chn_name_list, en_name_list

    def create_full_list(self):
        for dict1 in self.chn_name_list:
            for dict2 in self.en_name_list:
                if dict1[0] in dict2[1]:
                    print()

