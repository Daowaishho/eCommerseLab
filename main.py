import codecs
import re


def check_invalid_symbol(word):
    """
    :param word:输入的单词
    :return: boolean类型，是否属于制定的无效符号
    """
    symbol_set = ("…", "")
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


def extraction_gbk_to_utf8(input_path, output_path):
    """
    从源文件中提取
    :param input_path: 输入的文件路径
    :param output_path: 输出的文件路径
    :return:
    :
    """
    input_file = codecs.open(input_path, 'r', 'gbk')
    output_file = open(output_path, "w+")
    line = input_file.readline()
    recording = line.split('\t')
    for i in range(len(recording)):
        if i >= 4 and check_valid(recording[i]) is True:
            output_file.write(recording[i] + '\n')
    input_file.close()
    output_file.close()


# 配置
config = {
    'input_path': "/Users/huangshujie/PycharmProjects/pythonProject/raw_dataset/test",
    'output_path': "/Users/huangshujie/PycharmProjects/pythonProject/raw_dataset_after_cleaning/output_test"
}

if __name__ == "__main__":
    extraction_gbk_to_utf8(config['input_path'], config['output_path'])
