import codecs
import re
import hanlp
import time
import datetime
time_start = time.time()  # 记录开始时间
# function()   执行的程序
config = {
    'exclude_words': ("什么", "怎么", '图片', '视频', '教学', '一般'),
    'exclude_symbol': ("…", "", '\n', '\t', ''),
    'model': True,
    'line_number_limit': 30000,
    'input_path': "/Users/huangshujie/PycharmProjects/pythonProject/raw_dataset/user_tag_query.10W.TRAIN",
    'output_path': "/Users/huangshujie/PycharmProjects/pythonProject/raw_dataset_after_cleaning/output_test",
    'seed_keywords': ("微信",),
    'seed_relative_length': 0,
    'intermediary_words': [],
    'dataset_file': "/Users/huangshujie/PycharmProjects/pythonProject/raw_dataset_after_cleaning/output_test",
    'relative_words_file': '/Users/huangshujie/PycharmProjects/pythonProject/relative_words/relative_words',
    'word_frequency_file': '/Users/huangshujie/PycharmProjects/pythonProject/word_frequency_file/word_frequency_file',
    'intermediary_keywords': '/Users/huangshujie/PycharmProjects/pythonProject/word_frequency_file/word_frequency_file',
    'intermediary_relative_words_file': '/Users/huangshujie/PycharmProjects/pythonProject/intermediary_relative_words_file/intermediary_relative_words_file',
    'competitive_word_frequency_file': '/Users/huangshujie/PycharmProjects/pythonProject/competitive_word_frequency_file/competitive_word_frequency_file',
    'competitive_words_file': '/Users/huangshujie/PycharmProjects/pythonProject/competitive_keywords/competitive_keywords',
    'competitive_level_file': '/Users/huangshujie/PycharmProjects/pythonProject/competitive_level_file/competitive_level_file',
    'result': '/Users/huangshujie/PycharmProjects/pythonProject/result/result',
    'limit': 10,
    'get_all': False
}
weight_result = dict()
count = 0
word_frequency_dict = dict()
if config['model']:
    HanLP = hanlp.pipeline() \
        .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
        .append(hanlp.load('FINE_ELECTRA_SMALL_ZH'), output_key='tok') \
        .append(hanlp.load('CTB9_POS_ELECTRA_SMALL'), output_key='pos') \
        .append(hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH'), output_key='ner', input_key='tok') \
        .append(hanlp.load('CTB9_DEP_ELECTRA_SMALL', conll=0), output_key='dep', input_key='tok') \
        .append(hanlp.load('CTB9_CON_ELECTRA_SMALL'), output_key='con', input_key='tok')


result_file = open(config['result'], "a+")


def check_invalid_symbol(word):
    """
    :param word:输入的单词
    :return: boolean类型，是否属于制定的无效符号
    """
    symbol_set = config['exclude_symbol']
    return word in symbol_set


def meangingless_intermediary_keywords(word):
    """
    :param word:输入的单词
    :return: boolean类型，是否属于制定的无效的中介关键词
    """
    symbol_set = config['exclude_words']
    return word in symbol_set


def check_all_letters_or_numbers(word):
    """
    :param word:输入的单词
    :return: boolean类型，是否为长串的数字或者字母，以及网站
    """
    reg = r"[a-zA-Z0-9/:.．ａ-ｚＡ-Ｚ０-９]"
    pattern = re.compile(reg)
    return re.match(pattern, str(word)) is not None


def check_valid(word):
    """
    检验每一个单词是否合法
    :param word: 输入的单词
    :return: boolean类型，是否满足所有要求
    """
    if check_invalid_symbol(word) is True:
        return False
    if check_all_letters_or_numbers(word) is True:
        return False
    return True


def intermediary_keywords_cleaning(word):
    """
    去除单个关键词和种子关键词和数字
    :param word: 待检验的中介关键词
    :return: 是否合法,不合法返回False
    """
    if len(word) == 1:
        return False
    if word in config['seed_keywords']:
        return False
    if meangingless_intermediary_keywords(word):
        return False
    if config['seed_keywords'][0] in word:
        return False
    reg = r"[0-9０-９]"
    pattern = re.compile(reg)
    return re.match(pattern, str(word)) is None


def competitive_keywords_cleaning(word):
    """
    去除单个关键词和种子关键词和数字
    :param word: 待检验的中介关键词
    :return: 是否合法,不合法返回False
    """
    if len(word) == 1:
        return False
    if word in config['seed_keywords']:
        return False
    if word in config['intermediary_words']:
        return False
    if meangingless_intermediary_keywords(word):
        return False
    reg = r"[0-9０-９]"
    pattern = re.compile(reg)
    return re.match(pattern, str(word)) is None


def extraction_gbk_to_utf8(input_path, output_path):
    """
    从源文件中提取
    :param input_path: 输入的文件路径
    :param output_path: 输出的文件路径
    :return:
    :
    """
    input_file = codecs.open(input_path, 'r', 'gbk', errors='ignore')
    output_file = open(output_path, "w+")
    while True:
        line = input_file.readline()
        if line != '':
            recording = line.split('\t')
            for i in range(len(recording)):
                if i >= 4 and i != len(recording) - 1 and check_valid(recording[i]) is True:
                    output_file.write(recording[i] + '\n')
        else:
            break
    input_file.close()
    output_file.close()


def get_relative_words(seed_keywords, dataset_file, relative_words_file):
    input_file = codecs.open(dataset_file, 'r', 'utf8', errors='ignore')
    output_file = open(relative_words_file, "w+")
    line_count = 0
    while line_count < config['line_number_limit']:
        line = input_file.readline().split('\n')[0]
        line_count += 1
        if line != '':
            for words in seed_keywords:
                if words in line:
                    for seperated_words in HanLP(line)["tok"][0]:
                        output_file.write(seperated_words + '\\')
                    output_file.write('\n')
                    print_count()
                    break
        else:
            break
    input_file.close()
    output_file.close()


def get_word_frequency(relative_words_file, word_frequency_file, limit, get_all, mode = 0):
    global word_frequency_dict
    input_file = codecs.open(relative_words_file, 'r', 'utf8', errors='ignore')
    output_file = open(word_frequency_file, "w+")
    while True:
        line = input_file.readline()
        if line != '':
            words = line.split('\\')
            words.remove('\n')
            for word in words:
                if mode == 0:
                    if intermediary_keywords_cleaning(word) is True:
                        if word not in word_frequency_dict:  # 第一次出现的字符会被赋值1
                            word_frequency_dict[word] = 1
                        if word in word_frequency_dict:  # 再次出现的字符value加1
                            word_frequency_dict[word] += 1
                else:
                    if competitive_keywords_cleaning(word) is True:
                        if word not in word_frequency_dict:  # 第一次出现的字符会被赋值1
                            word_frequency_dict[word] = 1
                        if word in word_frequency_dict:  # 再次出现的字符value加1
                            word_frequency_dict[word] += 1
        else:
            break
    word_frequency_dict = sorted(word_frequency_dict.items(), key=lambda x: x[1], reverse=True)
    for key, value in word_frequency_dict:
        if get_all:
            output_file.write(key + ':' + str(value) + '\n')
        else:
            if limit > 0:
                output_file.write(key + ':' + str(value) + '\n')
                limit -= 1
    input_file.close()
    output_file.close()


def get_intermediary_words_from_file(intermediary_word_file):
    input_file = codecs.open(intermediary_word_file, 'r', 'utf8', errors='ignore')
    while True:
        line = input_file.readline().split(':')[0]
        if line != '':
            global config
            config['intermediary_words'].append(line)
        else:
            break
    input_file.close()


def calculate_weight(dataset_file):
    global config
    global weight_result
    seed_relative_num = 0
    input_file = codecs.open(dataset_file, 'r', 'utf8', errors='ignore')
    line_count = 0
    while line_count < config['line_number_limit']:
        line = input_file.readline().split('\n')[0]
        line_count += 1
        if line != '':
            if config['seed_keywords'][0] in line:
                seed_relative_num += 1
                for words in config['intermediary_words']:
                    if words in line:
                        if words not in weight_result:  # 第一次出现的字符会被赋值1
                            weight_result[words] = 1
                        if words in weight_result:  # 再次出现的字符value加1
                            weight_result[words] += 1
        else:
            break
    for words in config['intermediary_words']:
        weight_result[words] = weight_result[words] / seed_relative_num
    print(weight_result)
    input_file.close()


competitive_keywords_dict = {}

stop = 0


def get_competitive_keywords(dataset_file, competitive_words_file):
    global competitive_keywords_dict
    input_file = codecs.open(dataset_file, 'r', 'utf8', errors='ignore')
    output_file = open(competitive_words_file, "w+")
    line_count = 0
    while line_count < config['line_number_limit']:
        line = input_file.readline().split('\n')[0]
        line_count += 1
        if line != '':
            # have seed , doesn`t have key
            if config['seed_keywords'][0] not in line:
                for keywords in weight_result:
                    if keywords in line:
                        if keywords not in competitive_keywords_dict:
                            competitive_keywords_dict[keywords] = {}
                        for seperated_words in HanLP(line)["tok"][0]:
                            if seperated_words != keywords and competitive_keywords_cleaning(seperated_words) is True:
                                print("separate:"+str(seperated_words)+"\nkeywords:"+str(keywords)+"\nline:"+str(line))
                                if seperated_words not in competitive_keywords_dict[keywords]:  # 第一次出现的字符会被赋值1
                                    competitive_keywords_dict[keywords][seperated_words] = 1
                                if seperated_words in competitive_keywords_dict[keywords]:  # 再次出现的字符value加1
                                    competitive_keywords_dict[keywords][seperated_words] += 1
        else:
            break
    for sub_dict in competitive_keywords_dict:
        str1 = sub_dict.__str__()
        print(str1)
        sub_dict = sorted(competitive_keywords_dict[sub_dict].items(), key=lambda x: x[1], reverse=True)
        for key, value in sub_dict:
            output_file.write(str1+":"+key + ':' + str(value) + '\n')
            break
    input_file.close()
    output_file.close()


def get_competitive_level(data_set, competitive_keywords_file, competitive_level_file):
    input_file = codecs.open(competitive_keywords_file, 'r', 'utf8', errors='ignore')
    output_file = open(competitive_level_file, "w+")
    s_word = config['seed_keywords'][0]
    while True:
        line = input_file.readline()
        if line != '':
            words = line.split(':')
            k_word = words[1]
            a_word = words[0]
            ka = 0
            a = 0
            sa = 0
            data_set_file = codecs.open(data_set, 'r', 'utf8', errors='ignore')
            line_count = 0
            while line_count < config['line_number_limit']:
                record = data_set_file.readline()
                line_count += 1
                if record != '':
                    print_count()
                    if k_word in record and a_word in record:
                        ka += 1
                    if a_word in record:
                        a += 1
                    if s_word in record and a_word in record:
                        sa += 1
                    print(k_word + "ka:" + str(ka) + "a" + str(a) + "sa" + str(sa))
                else:
                    break
            output_file.write(k_word+":"+str(ka / (a - sa))+'\n')
            data_set_file.close()
        else:
            break
    input_file.close()
    output_file.close()


def print_competition(competitive_words_file):
    input_file = codecs.open(competitive_words_file, 'r', 'utf8', errors='ignore')
    temp_dict = {}
    while True:
        line = input_file.readline()
        if line != '' and line != '\n':
            words = line.split(':')
            temp_dict[words[0]] = words[1].replace('\n', '')
        else:
            break
    temp_dict = sorted(temp_dict.items(), key=lambda x: x[1], reverse=True)
    for key, value in temp_dict:
        print(key+"的竞争度为"+value)
        result_file.write(key+"的竞争度为"+value+'\n')
    input_file.close()


def print_count():
    global count
    print(count)
    count += 1


if __name__ == "__main__":
    # 数据清洗
    # extraction_gbk_to_utf8(config['input_path'], config['output_path'])
    # 给出种子关键字找出关联词并分词
    get_relative_words(config['seed_keywords'], config['dataset_file'], config['relative_words_file'])
    # 求出词频并选出limit个数的关键词
    get_word_frequency(config['relative_words_file'], config['word_frequency_file'], config['limit'], config['get_all'])
    # 把求出的关键词读入其中
    get_intermediary_words_from_file(config['intermediary_keywords'])
    # 计算权重
    calculate_weight(config['dataset_file'])
    get_competitive_keywords(config['dataset_file'], config['competitive_words_file'])
    get_competitive_level(config['dataset_file'], config['competitive_words_file'], config['competitive_level_file'])
    print_competition(config['competitive_level_file'])
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print("种子关键字"+config['seed_keywords'][0]+"记录条目数目"+str(config['line_number_limit'])+"\n用时："+str(time_sum)+"s")
    result_file.write("种子关键字:"+config['seed_keywords'][0]+"\n记录条目数目:"+str(config['line_number_limit'])+"\n用时："+str(time_sum) + "s\n")
    result_file.write("中介关键字个数:" + str(config['limit']) + '\n')
    current_time = datetime.datetime.now()
    result_file.write("time:    " + str(current_time)+'\n')
    result_file.write('-' * 100 + '\n')
    result_file.close()
