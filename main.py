import codecs
import re
import hanlp

config = {
    'model': False,
    'input_path': "/Users/huangshujie/PycharmProjects/pythonProject/raw_dataset/user_tag_query.10W.TRAIN",
    'output_path': "/Users/huangshujie/PycharmProjects/pythonProject/raw_dataset_after_cleaning/output_test",
    'seed_keywords': ("篮球",),
    'dataset_file': "/Users/huangshujie/PycharmProjects/pythonProject/raw_dataset_after_cleaning/output_test",
    'relative_words_file': '/Users/huangshujie/PycharmProjects/pythonProject/relative_words/relative_words',
    'word_frequency_file': '/Users/huangshujie/PycharmProjects/pythonProject/word_frequency_file/word_frequency_file',
    'limit': 10,
    'get_all': False
}
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


def check_invalid_symbol(word):
    """
    :param word:输入的单词
    :return: boolean类型，是否属于制定的无效符号
    """
    symbol_set = ("…", "", '\n', '\t', '')
    return word in symbol_set


def meangingless_intermediary_keywords(word):
    """
    :param word:输入的单词
    :return: boolean类型，是否属于制定的无效的中介关键词
    """
    symbol_set = ("什么", "怎么")
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
    while True:
        line = input_file.readline().split('\n')[0]
        print(line)
        if line != '':
            for words in seed_keywords:
                if words in line:
                    for seperated_words in HanLP(line)["tok"][0]:
                        output_file.write(seperated_words + '\\')
                    output_file.write('\n')
                    break
        else:
            break
    input_file.close()
    output_file.close()


def get_word_frequency(relative_words_file, word_frequency_file, limit, get_all):
    global word_frequency_dict
    input_file = codecs.open(relative_words_file, 'r', 'utf8', errors='ignore')
    output_file = open(word_frequency_file, "w+")
    while True:
        line = input_file.readline()
        if line != '':
            words = line.split('\\')
            words.remove('\n')
            for word in words:
                if intermediary_keywords_cleaning(word) is True:
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


# 配置


def print_count():
    global count
    print(count)
    count += 1


if __name__ == "__main__":
    # 数据清洗
    # extraction_gbk_to_utf8(config['input_path'], config['output_path'])
    # 给出种子关键字找出关联词并分词
    # get_relative_words(config['seed_keywords'], config['dataset_file'], config['relative_words_file'])
    # 求出词频
    get_word_frequency(config['relative_words_file'], config['word_frequency_file'], config['limit'], config['get_all'])
