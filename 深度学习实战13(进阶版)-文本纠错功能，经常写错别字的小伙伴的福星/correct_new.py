import random
from tqdm import tqdm
import string

# 生成一个随机数列表，用于选择进行操作的字符
def random_select_list(select_nums, lens_max):
    # 参数为要产生的随机数数量和上限
    random_index_list = []
    for _ in range(select_nums):
        while True:
            idx = random.randint(0, lens_max)
            if idx not in random_index_list:
                random_index_list.append(random.randint(0, lens_max))
                break
    return random_index_list

# 定义英文字母字典，用于序列内英文字符的替换
def get_en_char_list():
    en_char_list=[]
    for i in range(97,123):
        en_char_list.append(chr(i))
    for i in range(65,91):
        en_char_list.append(chr(i))
    return en_char_list

# 生成拼音-汉字、汉字-拼音字典
def get_dict(pinyin_file_path):
    # 读取拼音文件
    with open(pinyin_file_path, 'r', encoding='utf-8') as f:
        pinyin_list = f.readlines()

    # 创建四种类型的拼音-汉字字典
    unicode2pinyin = {}  # 汉字unicode为key，value为其拼音
    pinyin2unicode = {}  # key为拼音，value为对应汉字unicode
    char2pinyin = {}  # 汉字为key，value为其拼音
    pinyin2char = {}  # key为拼音，value为对应汉字

    # 经测试，unicode索引的方式可能会出错，不如直接用汉字进行索引
    for line in pinyin_list:
        # 原始数据格式为 'U+548C: hé,hè,hú,huó,huò,huo  # 和'
        # '#'后为对应的汉字
        line_list = line.split('#')
        ucode, pinyin = line_list[0].replace(' ', '').split(':')  # unicode编码及对应拼音str
        char = line_list[1].replace(' ', '').replace('\n','')  #对应汉字
        pinyin = pinyin.split(',')  # 处理为拼音list
        for y in pinyin:
            if y not in pinyin2unicode.keys():
                # 拼音-unicode dict
                # 同个读音下会有多个不同的unicode
                pinyin2unicode[y] = [ucode.replace('U+', r"\u").lower().encode('utf-8')]  # 对unicode的处理只是为了方便后期和输入文字的unicode进行对比，下方对unicode的处理也是如此
            else:pinyin2unicode[y].append(ucode.replace('U+', r"\u").lower().encode('utf-8'))
            if y not in pinyin2char.keys():
                # 拼音-汉字 dict
                # 同个读音下会有多个不同的汉字
                pinyin2char[y] = [char]
            else:pinyin2char[y].append(char)
        # unicode - 拼音 dict
        unicode2pinyin[ucode.replace('U+', r"\u").lower().encode('utf-8')] = pinyin
        # 汉字 - 拼音 dict
        char2pinyin[char] = pinyin
    return char2pinyin, pinyin2char

# 但是重复影响也不大，后期可去除
def get_single_pinyin(pinyin2char):
    # 找出仅有一个汉字的拼音
    return [pinyin for pinyin in pinyin2char.keys() if len(pinyin2char[pinyin])==1]

def modify_char(select_nums, text):
    # 修改指定个数的字符
    # 针对单字错误的，需按字符对原始文本进行切分,随机对其中的字符进行替换
    text_char_list = [c for c in text]
    random_index_list = random_select_list(select_nums, len(text_char_list)-1)
    for i, idx in enumerate(random_index_list):
        if (text_char_list[idx] not in char2pinyin.keys()) and (text_char_list[idx] not in en_char_list):
            while True:
                idx = random_select_list(1, len(text_char_list)-1)[0]
                if (idx not in random_index_list) and ((text_char_list[idx] in char2pinyin.keys()) or (text_char_list[idx] in en_char_list)):
                    random_index_list[i] = idx
                    break
    random_index_list.sort()
    for idx in random_index_list:
        tmp = text_char_list[idx]
        if text_char_list[idx] in en_char_list:
            # 英文字符随机替换为其他英文字符，暂不考虑键盘上的位置
            while True:
                select_idx = random_select_list(1, len(en_char_list)-1)[0]
                if (en_char_list[select_idx].lower()) != (text_char_list[idx].lower()):
                    text_char_list[idx] = en_char_list[select_idx]
                    break
        else:
            # 中文字符随机替换为其他同音汉字
            char_pinyin_list = char2pinyin[text_char_list[idx]]
            select_pinyin_idx = random_select_list(1, len(char_pinyin_list)-1)[0]
            select_pinyin = char_pinyin_list[select_pinyin_idx]
            char_list = pinyin2char[select_pinyin]
            while True:
                if len(char_list)==1:
                    # 表示该拼音对应只有一个汉字，直接返回，该汉字也可能和原来汉字相同。
                    text_char_list[idx] = char_list[0]
                    break
                select_char_idx = random_select_list(1, len(char_list)-1)[0]
                if char_list[select_char_idx] != text_char_list[idx]:
                    text_char_list[idx] = char_list[select_char_idx]
                    break
        # print(f'mmodify idx:{idx}\t{tmp}->{text_char_list[idx]}')
    return ''.join(text_char_list)+'\t'+text

def delete_char(select_nums, text):
    # 删除指定个数的字符
    text_char_list = [c for c in text]
    random_index_list = random_select_list(select_nums, len(text_char_list)-1)
    for i, idx in enumerate(random_index_list):
        if (text_char_list[idx] not in char2pinyin.keys()) and (text_char_list[idx] not in en_char_list):
            while True:
                idx = random_select_list(1, len(text_char_list)-1)[0]
                if (idx not in random_index_list) and ((text_char_list[idx] in char2pinyin.keys()) or (text_char_list[idx] in en_char_list)):
                    random_index_list[i] = idx
                    break
    random_index_list.sort()
    # for i in random_index_list:
        # print(f'delete idx:{i}--{text_char_list[i]}')
    text_char_list = [c for i ,c in enumerate(text_char_list) if i not in random_index_list]
    return ''.join(text_char_list)+'\t'+text


def add_char(select_nums, text):
    # 增加指定个数的字符
    text_char_list = [c for c in text]
    random_index_list = random_select_list(select_nums, len(text_char_list)-1)
    for i, idx in enumerate(random_index_list):
        if (text_char_list[idx] not in char2pinyin.keys()) and (text_char_list[idx] not in en_char_list):
            while True:
                idx = random_select_list(1, len(text_char_list)-1)[0]
                if (idx not in random_index_list) and ((text_char_list[idx] in char2pinyin.keys()) or (text_char_list[idx] in en_char_list)):
                    random_index_list[i] = idx
                    break
    random_index_list.sort()
    for i , idx in enumerate(random_index_list):
        if text_char_list[idx+i] in en_char_list:
            # 随机增加英文字符
            select_idx = random_select_list(1, len(en_char_list)-1)[0]
            text_char_list.insert(idx+i+1,en_char_list[select_idx])
        else:
            char_pinyin_list = char2pinyin[text_char_list[idx+i]]
            select_pinyin_idx = random_select_list(1, len(char_pinyin_list)-1)[0]
            select_pinyin = char_pinyin_list[select_pinyin_idx]
            char_list = pinyin2char[select_pinyin]
            select_char_idx = random_select_list(1, len(char_list)-1)[0]
            text_char_list.insert(idx+i+1,char_list[select_char_idx])
        # print(f'add idx:{idx+i+1}\t{text_char_list[idx+i+1]}')
    return ''.join(text_char_list)+'\t'+text

# 多条输入数据处理
def data_convert(sentence_list, convert_func, convert_nums, select_nums):
    # 输入文本序列，转换方式（修改、删除或增加），单句转换次数，单句内更改个数
    convert_sentence_list = []
    for sentence in tqdm(sentence_list):
        for _ in range(convert_nums):
            convert_sentence = convert_func(select_nums, sentence)
            if convert_sentence not in convert_sentence_list:
                convert_sentence_list.append(convert_sentence)
    return convert_sentence_list


pinyin_file_path = 'kHanyuPinlu.txt'
char2pinyin, pinyin2char = get_dict(pinyin_file_path)
en_char_list = get_en_char_list()
# 原始文本数据
sentence_list = [
    "关于张三认知的通知",
    "即成为原合同不可分割的组成部分",
    "与原合同具有同等法律效力",
    "除本合同明确修改的部分外",
    "其他服务条款均以原合同中约定的内容为准",
    "原合同应完全继续有效"
   ]

modify_sentence_list = data_convert(sentence_list, modify_char, 3, 1)
delete_sentence_list = data_convert(sentence_list,delete_char, 3, 1)
add_sentence_list = data_convert(sentence_list, add_char, 3, 1)

for sentence in modify_sentence_list:
    print(sentence)

# 判断该句是否仅为数字和标点符号
def word_not_include(sentence):
    punc = string.punctuation  # 英文标点符号
    pun = "，。……——“”‘’！；【】"  # 中文符号
    sentence_list = [c for c in sentence if (not c.isdigit() and c not in punc and c not in pun)]
    if len(sentence_list)==0:
        print(sentence)
        return True
    else: return False

# 通过读取文件获取句子列表，并对其中不符合要求的进行去除
def get_sentence_list(file_path):
    with open(file_path, 'r', encoding='utf-8')as f:
        sentence_list = f.readlines()
    sentence_list = [sentence.replace('\n', '') for sentence in sentence_list if not word_not_include(sentence.replace('\n',''))]
    return sentence_list

