import argparse
import datetime
import operator
from collections import defaultdict

import numpy as np
import pandas as pd
import regex


def list_to_dict(convert_list, data_type):
    add_dict = defaultdict(data_type)
    for key, value in convert_list:
        add_dict[key] += value
    return add_dict


def fill_blank_brand(dateframe, domain_list):
    for each_domain in domain_list:
        dateframe['standard_chn_name'][dateframe['standard_en_name'] == each_domain[1]] = each_domain[0]
        dateframe['standard_en_name'][dateframe['standard_chn_name'] == each_domain[0]] = each_domain[1]

    dateframe['standard_chn_name'][dateframe['standard_chn_name'].isnull()] = \
        dateframe['cln_chn_name'][dateframe['standard_chn_name'].isnull()]

    dateframe['standard_en_name'][dateframe['standard_en_name'].isnull()] = \
        dateframe['cln_en_name'][dateframe['standard_en_name'].isnull()]

    return dateframe


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
        trash_words_reg = ['国产', '授权', '行货', '原装', '海淘', '新西兰',
                           '法国', '香港', '港版', '澳洲', '英国',
                           '美国', '美版', '加拿大', '意大利', '德国',
                           '(荷兰(?!朵))', '日本', '韩国']

        self.pattern_replace_pair_list = [('|'.join(trash_words_reg), r'')]


class WordReplacer(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = [(r'の', r'之')]


class BrandSplit(BaseReplacer):
    def transform(self, text):
        chn_brand = regex.findall(r'[\u4e00-\u9fff]+', text)
        en_brand = regex.findall(r'[a-z]+', text)
        en_brand = [regex.sub(pattern=r'^na$', repl=r'', string=string) for string in en_brand]
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

        self.frame = self.get_frame()

    def clean_brand(self):

        print('Processing Brand Cleaning ... ')

        processors = [TrashWords(), WordReplacer(), LowerCaseConverter(), BrandSplit()]
        for processor in processors:
            self.brands = list(map(processor.transform, self.brands))
        return self.brands

    def get_frame(self):
        self.new_brands = self.clean_brand()
        mapping_frame = pd.DataFrame(self.new_brands, index=list(self.count_list.index))
        mapping_frame['counts'] = self.count_list
        mapping_frame[args.column] = mapping_frame.index
        mapping_frame.rename(columns={0: 'cln_chn_name', 1: 'cln_en_name'}, inplace=True)

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
        en_name_dict = en_name_dict

        return chn_name_list, en_name_dict

    def create_full_list(self):

        self.chn_name_list, self.en_name_dict = self.merge_brand()

        full_list = []

        for chn_name, en_name in self.chn_name_list:
            single_list = set()
            for each_en_name in en_name:
                single_list = single_list.union(self.en_name_dict[each_en_name])
            single_list = single_list | set(en_name)
            full_list.append(single_list)

        list_length = len(full_list)

        for set_index1 in range(list_length):
            for set_index2 in range(set_index1 + 1, list_length):
                if full_list[set_index1] & full_list[set_index2]:
                    full_list[set_index1] |= full_list[set_index2]
                    full_list[set_index2] = set()

        full_list = sorted(full_list, key=lambda x: -len(x))
        full_list = filter(lambda x: len(x) > 0, full_list)

        return full_list


class DomainBrandMap:
    def __init__(self, full_list, mapping_frame):
        self.full_list = full_list
        self.mapping_frame = mapping_frame

    def create_counts_dict(self):
        chn_counts_list = list(zip(self.mapping_frame.ix[:, 0], self.mapping_frame['counts']))
        en_counts_list = list(zip(self.mapping_frame.ix[:, 1], self.mapping_frame['counts']))

        chn_counts_dict = list_to_dict(chn_counts_list, int)
        en_counts_dict = list_to_dict(en_counts_list, int)

        return chn_counts_dict, en_counts_dict

    def counts_map(self):

        self.chn_counts_dict, self.en_counts_dict = self.create_counts_dict()

        self.brand_list = []
        for each_brand in self.full_list:
            if len(each_brand) > 2:
                self.each_brand_list = {}
                for name in each_brand:
                    if regex.findall(r'[a-z]', name):
                        temp_dict = dict([(name, self.en_counts_dict[name])])
                    else:
                        temp_dict = dict([(name, self.chn_counts_dict[name])])
                    self.each_brand_list.update(temp_dict)
                self.brand_list.append(self.each_brand_list)
            else:
                self.brand_list.append({list(each_brand)[0]: 1, list(each_brand)[1]: 1})

        self.sorted_brand_list = [sorted(each_dict.items(), key=lambda k: k[1], reverse=True) for each_dict in
                                  self.brand_list]
        self.brand_key = [sorted(each_dict.keys(), key=lambda k: k[0], reverse=True) for each_dict in self.brand_list]

        return self.sorted_brand_list

    def map_domain_name(self):

        domain_name_list = []
        self.counts_map_list = self.counts_map()

        for each_brand in self.counts_map_list:

            each_brand_list = []

            if len(each_brand) > 2:
                for name_fre in each_brand:
                    if regex.findall(r'[\u4e00-\u9fff]', name_fre[0]) and len(each_brand_list) == 0:
                        each_brand_list.append(name_fre[0])
                        break
                for name_fre in each_brand:
                    if regex.findall(r'[a-z]', name_fre[0]) and len(each_brand_list) == 1:
                        each_brand_list.append(name_fre[0])
                        break
            else:
                each_brand_list += (sorted(map(lambda x: x[0], each_brand), reverse=True))

            domain_name_list.append(each_brand_list)

        mapping_dict = defaultdict(list)

        for list_index, each_brand in enumerate(self.brand_key):

            if len(each_brand) > 2:
                for name in each_brand:
                    if regex.findall(r'[\u4e00-\u9fff]', name):
                        mapping_dict[domain_name_list[list_index][0]].append(name)
                    else:
                        mapping_dict[domain_name_list[list_index][1]].append(name)
            else:
                for name in each_brand:
                    mapping_dict[name].append(name)

        brand_mapping_frame = pd.DataFrame.from_dict(mapping_dict, orient='index')
        brand_mapping_frame['name_found'] = (brand_mapping_frame.notnull()).astype(int).sum(axis=1)
        brand_mapping_frame = brand_mapping_frame.sort_values('name_found', axis=0, ascending=False)
        brand_mapping_frame.drop('name_found', 1, inplace=True)

        print('Saving Mapping DataFrame ... ')
        brand_mapping_frame.to_csv(regex.sub(r'(/.+)((\.csv$)|(\.txt$))', r'', args.input) +
                                   '/brand_mapping_frame_%s.csv' % datetime.datetime.now().strftime("%Y_%m_%d"), index=False)

        brand_mapping_frame['standard_brand'] = brand_mapping_frame.index

        return brand_mapping_frame, domain_name_list


def main():

    AllData = pd.read_csv(args.input)
    brand_list = BrandSeparator(AllData[args.column])
    brand_full_list = brand_list.create_full_list()
    brand_list_frame = brand_list.frame

    domain_frame, domain_list = DomainBrandMap(brand_full_list, brand_list_frame).map_domain_name()
    domain_frame = pd.melt(domain_frame, id_vars=['standard_brand'], var_name='number').drop('number', 1)

    AllData = AllData.merge(brand_list_frame[[args.column, 'cln_chn_name', 'cln_en_name']], how='left', on=args.column)
    AllData = AllData.merge(domain_frame, how='left', left_on='cln_chn_name', right_on='value').drop('value', 1)
    AllData = AllData.merge(domain_frame, how='left', left_on='cln_en_name', right_on='value').drop('value', 1)
    AllData.rename(columns={'standard_brand_x': 'standard_chn_name', 'standard_brand_y': 'standard_en_name'}, inplace=True)
    print('Filling Blank Brand Names ...')
    AllData = fill_blank_brand(AllData, domain_list)

    AllData.to_csv(regex.sub(r'(/.+)((\.csv$)|(\.txt$))', r'', args.input) + '/brand_standard_output_%s.csv'
                   % datetime.datetime.now().strftime("%Y_%m_%d"), index=False)

    print('Done ... ')

    return AllData


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Brand-Standard')
    parser.add_argument('--input', type=str, help='Input DataFrame', )
    parser.add_argument('--column', type=str, help='Brand Column Name')
    parser.add_argument('--mapping', action='store_true', help='Standard from mapping table')
    # parser.add_argument('--', type=int, help='specify fold')

    args = parser.parse_args()
    main()
