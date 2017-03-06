#! -*- coding:utf-8 -*-
import sys
import jieba
import os
import string
from collections import defaultdict
import pandas as pd
import Levenshtein
import lzma
import csv
import re
import numpy as np



os.chdir('C:/Users/xianxian/Desktop/jieba/')
jieba.load_userdict("jieba.dict.txt")
skuname = pd.read_csv('skuname.csv')
sku_std = pd.read_csv('spu.csv')

pattern1 = '韩国|进口|限量版|全国|定做|配送|同城|速递|预定|北京|天津|上海|重庆|合肥|宿州|淮北|阜阳|蚌埠|淮南|滁州|马鞍山|芜湖|铜陵|安庆|黄山|六安|池州|宣城|亳州|界首|明光|天长|桐城|宁国|巢湖|厦门|福州|南平|三明|莆田|泉州|漳州|龙岩|宁德|福清|长乐|邵武|武夷山|建瓯|建阳|永安|石狮|晋江|南安|龙海|漳平|福安|福鼎|兰州|嘉峪关|金昌|白银|天水|酒泉|张掖|武威|庆阳|平凉|定西|陇南|玉门|敦煌|临夏|合作|广州|深圳|清远|韶关|河源|梅州|潮州|汕头|揭阳|汕尾|惠州|东莞|珠海|中山|江门|佛山|肇庆|云浮|阳江|茂名|湛江|从化|增城|英德|连州|乐昌|南雄|兴宁|普宁|陆丰|恩平|台山|开平|鹤山|高要|四会|罗定|阳春|化州|信宜|高州|吴川|廉江|雷州|贵阳|六盘水|遵义|安顺|毕节|铜仁|清镇|赤水|仁怀|凯里|都匀|兴义|福泉|石家庄|邯郸|唐山|保定|秦皇岛|邢台|张家口|承德|沧州|廊坊|衡水|辛集|藁城|晋州|新乐|鹿泉|遵化|迁安|霸州|三河|定州|涿州|安国|高碑店|泊头|任丘|黄骅|河间|冀州|深州|南宫|沙河|武安|哈尔滨|齐齐哈尔|黑河|大庆|伊春|鹤岗|佳木斯|双鸭山|七台河|鸡西|牡丹江|绥化|双城|尚志|五常|阿城|讷河|北安|五大连池|铁力|同江|富锦|虎林|密山|绥芬河|海林|宁安|安达|肇东|海伦|郑州|开封|洛阳|平顶山|安阳|鹤壁|新乡|焦作|濮阳|许昌|漯河|三门峡|南阳|商丘|周口|驻马店|信阳|济源|巩义|邓州|永城|汝州|荥阳|新郑|登封|新密|偃师|孟州|沁阳|卫辉|辉县|林州|禹州|长葛|舞钢|义马|灵宝|项城|武汉|十堰|襄樊|荆门|孝感|黄冈|鄂州|黄石|咸宁|荆州|宜昌|随州|仙桃|天门|潜江|丹江口|老河口|枣阳|宜城|钟祥|汉川|应城|安陆|广水|麻城|武穴|大冶|赤壁|石首|洪湖|松滋|宜都|枝江|当阳|恩施|利川|长沙|衡阳|张家界|常德|益阳|岳阳|株洲|湘潭|郴州|永州|邵阳|怀化|娄底|耒阳|常宁|浏阳|津市|沅江|汨罗|临湘|醴陵|湘乡|韶山|资兴|武冈|洪江|冷水江|涟源|吉首|长春|吉林市|白城|松原|四平|辽源|通化|白山|德惠|九台|榆树|磐石|蛟河|桦甸|舒兰|洮南|大安|双辽|公主岭|梅河口|集安|临江|延吉|图们|敦化|珲春|龙井|和龙|南昌|九江|景德镇|鹰潭|新余|萍乡|赣州|上饶|抚州|宜春|吉安|瑞昌|乐平|瑞金|南康|德兴|丰城|樟树|高安|井冈山|贵溪|南京|徐州|连云港|宿迁|淮安|盐城|扬州|泰州|南通|镇江|常州|无锡|苏州|江阴|宜兴|邳州|新沂|金坛|溧阳|常熟|张家港|太仓|昆山|吴江|如皋|海门|启东|大丰|东台|高邮|仪征|扬中|句容|丹阳|兴化|姜堰|泰兴|靖江|沈阳|大连|朝阳|阜新|铁岭|抚顺|本溪|辽阳|鞍山|丹东|营口|盘锦|锦州|葫芦岛|新民|瓦房店|普兰店|庄河|北票|凌源|调兵山|开原|灯塔|海城|凤城|东港|大石桥|盖州|凌海|北宁|兴城|济南|青岛|聊城|德州|东营|淄博|潍坊|烟台|威海|日照|临沂|枣庄|济宁|泰安|莱芜|滨州|菏泽|章丘|胶州|胶南|即墨|平度|莱西|临清|乐陵|禹城|安丘|昌邑|高密|青州|诸城|寿光|栖霞|海阳|龙口|莱阳|莱州|蓬莱|招远|文登|荣成|乳山|滕州|曲阜|兖州|邹城|新泰|肥城|西安|延安|铜川|渭南|咸阳|宝鸡|汉中|榆林|商洛|安康|韩城|华阴|兴平|太原|大同|朔州|阳泉|长治|晋城|忻州|吕梁|晋中|临汾|运城|古交|潞城|高平|原平|孝义|汾阳|介休|侯马|霍州|永济|河津|成都|广元|绵阳|德阳|南充|广安|遂宁|内江|乐山|自贡|泸州|宜宾|攀枝花|巴中|达州|资阳|眉山|雅安|崇州|邛崃|都江堰|彭州|江油|什邡|广汉|绵竹|阆中|华蓥|峨眉山|万源|简阳|西昌|昆明|曲靖|玉溪|丽江|昭通|思茅|临沧|保山|安宁|宣威|芒市|瑞丽|大理|楚雄|个旧|开远|景洪|杭州|宁波|湖州|嘉兴|舟山|绍兴|衢州|金华|台州|温州|丽水|临安|富阳|建德|慈溪|余姚|奉化|平湖|海宁|桐乡|诸暨|上虞|嵊州|江山|兰溪|永康|义乌|东阳|临海|温岭|瑞安|乐清|龙泉|西宁|格尔木|德令哈|海口市|三亚市|文昌市|琼海市|万宁市|东方市|儋州市|五指山市|南宁|桂林|柳州|梧州|贵港|玉林|钦州|北海|防城港|崇左|百色|河池|来宾|贺州|岑溪|桂平|北流|东兴|凭祥|宜州|合山|呼和浩特|包头|乌海|赤峰|呼伦贝尔|通辽|乌兰察布|鄂尔多斯|巴彦淖尔|满洲里|扎兰屯|牙克石|根河|额尔古纳|乌兰浩特|阿尔山|霍林郭勒|锡林浩特|二连浩特|丰镇|银川|石嘴山|吴忠|中卫|固原|灵武|青铜峡|拉萨|日喀则|乌鲁木齐|克拉玛依|石河子|阿拉尔|图木舒克|五家渠|北屯|喀什|阿克苏|和田|吐鲁番|哈密|阿图什|博乐|昌吉|阜康|米泉|库尔勒|伊宁|奎屯|塔城|乌苏|阿勒泰|香港|澳门'
pattern2 = '[【({<（《^】)}>》）*】)}>》）\[\]]|[0-9一二三四五六七八九十壹貳叁肆伍陸柒捌玖拾单]+(套|瓶|盒|支|箱|包|罐|听|件|袋|片|桶|瓶|个|只|装|ml|ML|g|kg|KG|G|克|毫升|升)'
l = [pattern2, pattern1]
pattern = '|'.join(l)


seg_list = []
for sku in skuname.skuName:
    seg_list.append(list(jieba.cut(re.sub(pattern, '', sku))))
'''
seg_list = []
for sku in skuname.skuName:
    seg_list.append(list(set(jieba.cut(re.sub(pattern, '', sku)))))

x = '魔都生活网超 可口可乐1.25L/瓶+美汁源果粒橙1.25L/瓶 分享装 可口可乐出品'
seg_list = list(jieba.cut(re.sub(pattern, '', x)))
'''
std_list = []
for std in sku_std.spu:
    std_list.append(list(jieba.cut(std)))
#-------------------------------------------------------------------------------------------------------

MISSING_VALUE_NUMERIC = -1.
def _unigrams(words):
    assert type(words) == list
    return words

'''
def _bigrams(words, join_string, skip=0):
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for k in range(1,skip+2):
                if i+k < L:
                    lst.append( join_string.join([words[i], words[i+k]]) )
    else:
        # set it as unigram
        lst = _unigrams(words)
    return lst
'''

def _edit_dist(str1, str2):
    try:
        # very fast
        # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
        # d = Levenshtein.ratio(str1, str2)
        d = Levenshtein.distance(str1, str2)/float(max(len(str1),len(str2)))
    except:
        # https://docs.python.org/2/library/difflib.html
        d = 1. - SequenceMatcher(lambda x: x==" ", str1, str2).ratio()
    return d

def _is_str_match(str1, str2, threshold=0.5):
    assert threshold >= 0.0 and threshold <= 1.0, "Wrong threshold."
    if float(threshold) == 1.0:
        return str1 == str2
    else:
        return (1. - _edit_dist(str1, str2)) >= threshold

def transform_one_tf(obs_tokens, target_tokens, str_match_threshold):
    obs_ngrams = _unigrams(obs_tokens)
    target_ngrams = _unigrams(target_tokens)
    val_list = []
    for w1 in obs_ngrams:
        s = 0.
        for w2 in target_ngrams:
            if _is_str_match(w1, w2, str_match_threshold):
                s += 1.
        val_list.append(s)
    if len(val_list) == 0:
        val_list = [MISSING_VALUE_NUMERIC]
    return val_list


def _try_divide(x, y, val=0.0):
    if y != 0.0:
        val = float(x) / y
    return val


def transform_one_norm_tf(obs_tokens, target_tokens, str_match_threshold):
    obs_ngrams = _unigrams(obs_tokens)
    target_ngrams = _unigrams(target_tokens)
    val_list = []
    for w1 in obs_ngrams:
        s = 0.
        for w2 in target_ngrams:
            if _is_str_match(w1, w2, str_match_threshold):
                s += 1.
                val_list.append(_try_divide(s, len(target_ngrams)))
    if len(val_list) == 0:
        val_list = [MISSING_VALUE_NUMERIC]
    return val_list
# ------------------------------------------------------------------------

# ------------------------------ TFIDF -----------------------------------
STR_MATCH_THRESHOLD = 0.85

d = defaultdict(lambda: 1)
for target in std_list:
    target_ngrams = _unigrams(target)
    for w in set(target_ngrams):
        d[w] += 1

def _get_idf(word):
    return np.log((len(std_list) - d[word] + 0.5) / (d[word] + 0.5))

def transform_one_tfidf(obs_tokens, target_tokens, str_match_threshold):
    obs_ngrams = _unigrams(obs_tokens)
    target_ngrams = _unigrams(target_tokens)
    val_list = []
    for w1 in obs_ngrams:
        s = 0.
        for w2 in target_ngrams:
            if _is_str_match(w1, w2, str_match_threshold):
                s += 1.
        val_list.append(s * _get_idf(w1))
    if len(val_list) == 0:
        val_list = [MISSING_VALUE_NUMERIC]
    return val_list
#---------------------------------------------------------------------------------------------------------------------------
#norm tfidf
def transform_one_norm_tfidf(obs_tokens, target_tokens, str_match_threshold):
    obs_ngrams = _unigrams(obs_tokens)
    target_ngrams = _unigrams(target_tokens)
    val_list = []
    for w1 in obs_ngrams:
        s = 0.
        for w2 in target_ngrams:
            if _is_str_match(w1, w2, str_match_threshold):
                s += 1.
        val_list.append(_try_divide(s, len(target_ngrams)) * _get_idf(w1))
    if len(val_list) == 0:
        val_list = [MISSING_VALUE_NUMERIC]
    return val_list

#bm25
lst = []
for target in std_list[0:3]:
    target_ngrams = _unigrams(target)
    lst.append(len(target_ngrams))
avg_ngram_doc_len = np.mean(lst)#返回research file平均每条有几个词



def transform_one_bm25(obs_tokens, target_tokens, str_match_threshold, BM25_B = 0.75, BM25_K1 = 1.6):
    obs_ngrams = _unigrams(obs_tokens)
    target_ngrams = _unigrams(target_tokens)
    K = BM25_K1 * (1 - BM25_B + BM25_B * _try_divide(len(target_ngrams), avg_ngram_doc_len))#代表对这条std_spu长度的考量, b越大对文档长度的惩罚越大,k1越大词频的作用越大
    val_list = []
    for w1 in obs_ngrams:
        s = 0.
        for w2 in target_ngrams:
            if _is_str_match(w1, w2, str_match_threshold):
                s += 1.
        bm25 = s * _get_idf(w1) * _try_divide(1 + BM25_K1, s + K)
        val_list.append(bm25)
    if len(val_list) == 0:
        val_list = [MISSING_VALUE_NUMERIC]
    return val_list



str_match_threshold = 0.75
all = []
for sku in seg_list:
    tf = []
    for std in std_list:
        tf.append(sum(transform_one_tf(sku, std, str_match_threshold)))
    norm_tf = []
    for std in std_list:
        norm_tf.append(sum(transform_one_norm_tf(sku, std, str_match_threshold)))
    tfidf = []
    for std in std_list:
        tfidf.append(sum(transform_one_tfidf(sku, std, str_match_threshold)))
    norm_tfidf = []
    for std in std_list:
        norm_tfidf.append(sum(transform_one_norm_tfidf(sku, std, str_match_threshold)))
    bm25 = []
    for std in std_list:
        bm25.append(sum(transform_one_bm25(sku, std, str_match_threshold)))
    all.append([sku, std_list[tf.index(max(tf))], max(tf), std_list[norm_tf.index(max(norm_tf))], max(norm_tf), std_list[tfidf.index(max(tfidf))], max(tfidf), std_list[norm_tfidf.index(max(norm_tfidf))], max(norm_tfidf),std_list[bm25.index(max(bm25))], max(bm25)])

df = pd.DataFrame(all)
df.columns = ['sku', 'std_tf', 'std_tf_score', 'std_norm_tf', 'std_norm_tf_score', 'std_tfidf', 'std_tfidf_score', 'std_norm_tfidf', 'std_norm_tfidf_score', 'std_bm25', 'std_bm25_score']
print(df.head())
df.to_csv('sku_tf_bm25.test.csv', sep=',', encoding='gb18030', index=False)




