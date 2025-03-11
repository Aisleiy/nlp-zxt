import os
import re
import math
import time
import jieba
import string
import nltk
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)
nltk.download('gutenberg', force=True)
nltk.download('punkt_tab', force=True)

# 设置 nltk 数据路径
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# ---------------------------
# 相关文件路径设置
# ---------------------------
chinese_corpus_train = os.path.join(os.getcwd(), 'wiki_zh', 'AA')
chinese_corpus_test = os.path.join(os.getcwd(), 'wiki_zh', 'AG')
stopwords_chinese_path = os.path.join(os.getcwd(), 'cn_stopwords.txt')

print("训练集路径:", chinese_corpus_train)
print("测试集路径:", chinese_corpus_test)

# ---------------------------
# 中文语料库数据加载
# ---------------------------
def loaddata_chinese_custom(train_dir, test_dir, mode):
    train_data = ""
    test_data = ""
    
    # 检查训练集目录
    if not os.path.exists(train_dir):
        print("训练集目录不存在。")
        return [], []
    
    # 读取训练集
    print("开始读取训练集文件...")
    for file_name in sorted(os.listdir(train_dir)):
        path = os.path.join(train_dir, file_name)
        if os.path.isfile(path) and file_name.endswith('.txt'):
            print("读取训练集文件:", file_name)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if not content:
                    print("训练集文件", file_name, "为空。")
                    continue
                train_data += content
            except Exception as e:
                print("读取训练集文件", file_name, "时出错:", e)
    
    # 检查测试集目录
    if not os.path.exists(test_dir):
        print("测试集目录不存在。")
        return [], []
    
    # 读取测试集
    print("开始读取测试集文件...")
    for file_name in sorted(os.listdir(test_dir)):
        if file_name.endswith('.txt') and file_name.startswith('wiki_'):
            num_part = file_name.replace('wiki_', '').replace('.txt', '')
            try:
                num = int(num_part)
                if 0 <= num <= 10:
                    path = os.path.join(test_dir, file_name)
                    print("读取测试集文件:", file_name)
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if not content:
                        print("测试集文件", file_name, "为空。")
                        continue
                    test_data += content
            except ValueError:
                continue
            except Exception as e:
                print("读取测试集文件", file_name, "时出错:", e)
    
    # 预处理：过滤非中文字符
    print("开始预处理训练集数据...")
    pattern = r'[^\u4e00-\u9fff]'  # 仅保留中文字符
    train_data = re.sub(pattern, '', train_data)
    test_data = re.sub(pattern, '', test_data)
    
    print("预处理后训练集数据长度:", len(train_data))
    print("预处理后测试集数据长度:", len(test_data))
    
    # 分词处理
    if mode == 'token':
        print("开始分词处理...")
        train_tokens = list(jieba.cut(train_data))
        test_tokens = list(jieba.cut(test_data))
        print("分词后的训练集tokens样本:", train_tokens[:10])
        print("分词后的测试集tokens样本:", test_tokens[:10])
    elif mode == 'char':
        train_tokens = list(train_data)
        test_tokens = list(test_data)
    else:
        raise ValueError("mode 参数只能为 'token' 或 'char'")
    
    # 加载停用词
    if not os.path.exists(stopwords_chinese_path):
        print("停用词文件不存在。")
        cn_stopwords = []
    else:
        with open(stopwords_chinese_path, 'r', encoding='utf-8') as f:
            cn_stopwords = [line.strip() for line in f.readlines()]
        print("停用词数量:", len(cn_stopwords))
    
    # 过滤停用词和非中文字符
    pattern_ch = re.compile(r'^[\u4e00-\u9fff]+$')
    train_tokens = [word for word in train_tokens if pattern_ch.match(word) and word not in cn_stopwords]
    test_tokens = [word for word in test_tokens if pattern_ch.match(word) and word not in cn_stopwords]
    
    print("过滤后训练集tokens数目:", len(train_tokens))
    print("过滤后测试集tokens数目:", len(test_tokens))
    
    return train_tokens, test_tokens

# ---------------------------
# 英文语料库数据加载
# ---------------------------
def loaddata_english_train(mode):
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('gutenberg', quiet=True)
    
    fileid_to_skip = 'austen-emma.txt'
    data = ""
    for fileid in gutenberg.fileids():
        if fileid == fileid_to_skip:
            continue
        data += gutenberg.raw(fileid)
    data = data.lower()
    
    if mode == 'token':
        tokens = word_tokenize(data)
        tokens = [word for word in tokens if word not in string.punctuation]
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    elif mode == 'char':
        tokens = list(data)
        tokens = [ch for ch in tokens if ch.isalpha()]
    else:
        raise ValueError("mode 参数只能为 'token' 或 'char'")
    return tokens

def loaddata_english_test(mode):
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('gutenberg', quiet=True)
    
    fileid = 'austen-emma.txt'
    data = gutenberg.raw(fileid).lower()
    
    if mode == 'token':
        tokens = word_tokenize(data)
        tokens = [word for word in tokens if word not in string.punctuation]
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    elif mode == 'char':
        tokens = list(data)
        tokens = [ch for ch in tokens if ch.isalpha()]
    else:
        raise ValueError("mode 参数只能为 'token' 或 'char'")
    return tokens

# ---------------------------
# 词频统计与信息熵计算函数
# ---------------------------
def get_tf(tf_dic, words):
    for word in words:
        tf_dic[word] = tf_dic.get(word, 0) + 1

def get_bigram_tf(tf_dic, words):
    for i in range(len(words)-1):
        bigram = (words[i], words[i+1])
        tf_dic[bigram] = tf_dic.get(bigram, 0) + 1

def get_trigram_tf(tf_dic, words):
    for i in range(len(words)-2):
        trigram = ((words[i], words[i+1]), words[i+2])
        tf_dic[trigram] = tf_dic.get(trigram, 0) + 1

def calculate_unigram_entropy(tf_dic_train, tf_dic_test):
    begin = time.time()
    total_train = sum(tf_dic_train.values())
    total_test = sum(tf_dic_test.values())
    
    if total_train == 0 or total_test == 0:
        print("无法计算一元模型信息熵，因为训练集或测试集数据为空。")
        return 0.0
    
    entropy = 0
    for word, count in tf_dic_test.items():
        jp = count / total_test
        freq_train = tf_dic_train.get(word, 0)
        if freq_train == 0:
            freq_train = 1  # 平滑处理
        cp = freq_train / total_train
        entropy += -jp * math.log(cp, 2)
    end = time.time()
    print("一元模型信息熵为：{:.6f} 比特/(词或字)，运行时间：{:.6f} s".format(entropy, end - begin))
    return entropy

def calculate_bigram_entropy(tf_dic_train, bigram_tf_dic_train, bigram_tf_dic_test):
    begin = time.time()
    total_bi_test = sum(bigram_tf_dic_test.values())
    
    if total_bi_test == 0:
        print("无法计算二元模型信息熵，因为测试集二元组数据为空。")
        return 0.0
    
    entropy = 0
    for bigram, count in bigram_tf_dic_test.items():
        jp = count / total_bi_test
        freq_bigram_train = bigram_tf_dic_train.get(bigram, 0)
        first_word_freq = tf_dic_train.get(bigram[0], 0)
        if freq_bigram_train == 0:
            freq_bigram_train = 1  # 平滑处理
        if first_word_freq == 0:
            first_word_freq = 1    # 平滑处理
        cp = freq_bigram_train / first_word_freq
        entropy += -jp * math.log(cp, 2)
    end = time.time()
    print("二元模型信息熵为：{:.6f} 比特/(词或字)，运行时间：{:.6f} s".format(entropy, end - begin))
    return entropy

def calculate_trigram_entropy(bigram_tf_dic_train, trigram_tf_dic_train, trigram_tf_dic_test):
    begin = time.time()
    total_tri_test = sum(trigram_tf_dic_test.values())
    
    if total_tri_test == 0:
        print("无法计算三元模型信息熵，因为测试集三元组数据为空。")
        return 0.0
    
    entropy = 0
    for trigram, count in trigram_tf_dic_test.items():
        jp = count / total_tri_test
        freq_trigram_train = trigram_tf_dic_train.get(trigram, 0)
        first_bigram_freq = bigram_tf_dic_train.get(trigram[0], 0)
        if freq_trigram_train == 0:
            freq_trigram_train = 1  # 平滑处理
        if first_bigram_freq == 0:
            first_bigram_freq = 1    # 平滑处理
        cp = freq_trigram_train / first_bigram_freq
        entropy += -jp * math.log(cp, 2)
    end = time.time()
    print("三元模型信息熵为：{:.6f} 比特/(词或字)，运行时间：{:.6f} s".format(entropy, end - begin))
    return entropy

# ---------------------------
# 主程序入口
# ---------------------------
def main():
    # 中文语料实验
    print("========== 中文语料实验 ==========")
    for mode in ['token', 'char']:
        print("\n【中文 - 模式：{}】".format(mode))
        try:
            train_tokens, test_tokens = loaddata_chinese_custom(chinese_corpus_train, chinese_corpus_test, mode)
            print("训练集tokens数目：{}, 测试集tokens数目：{}".format(len(train_tokens), len(test_tokens)))
            
            if not train_tokens or not test_tokens:
                print("训练集或测试集数据为空，跳过信息熵计算。")
                continue
            
            tf_dic_train = {}
            bigram_tf_dic_train = {}
            trigram_tf_dic_train = {}
            tf_dic_test = {}
            bigram_tf_dic_test = {}
            trigram_tf_dic_test = {}
            
            get_tf(tf_dic_train, train_tokens)
            get_bigram_tf(bigram_tf_dic_train, train_tokens)
            get_trigram_tf(trigram_tf_dic_train, train_tokens)
            
            get_tf(tf_dic_test, test_tokens)
            get_bigram_tf(bigram_tf_dic_test, test_tokens)
            get_trigram_tf(trigram_tf_dic_test, test_tokens)
            
            print("\n中文一元模型信息熵：")
            calculate_unigram_entropy(tf_dic_train, tf_dic_test)
            print("\n中文二元模型信息熵：")
            calculate_bigram_entropy(tf_dic_train, bigram_tf_dic_train, bigram_tf_dic_test)
            print("\n中文三元模型信息熵：")
            calculate_trigram_entropy(bigram_tf_dic_train, trigram_tf_dic_train, trigram_tf_dic_test)
        except Exception as e:
            print(f"运行时错误：{e}")
    
    # 英文语料实验
    print("\n========== 英文语料实验 ==========")
    for mode in ['token', 'char']:
        print("\n【英文 - 模式：{}】".format(mode))
        try:
            english_train = loaddata_english_train(mode)
            english_test = loaddata_english_test(mode)
            print("训练集tokens数目：{}, 测试集tokens数目：{}".format(len(english_train), len(english_test)))
            
            if not english_train or not english_test:
                print("训练集或测试集数据为空，跳过信息熵计算。")
                continue
            
            tf_dic_train = {}
            bigram_tf_dic_train = {}
            trigram_tf_dic_train = {}
            tf_dic_test = {}
            bigram_tf_dic_test = {}
            trigram_tf_dic_test = {}
            
            get_tf(tf_dic_train, english_train)
            get_bigram_tf(bigram_tf_dic_train, english_train)
            get_trigram_tf(trigram_tf_dic_train, english_train)
            
            get_tf(tf_dic_test, english_test)
            get_bigram_tf(bigram_tf_dic_test, english_test)
            get_trigram_tf(trigram_tf_dic_test, english_test)
            
            print("\n英文一元模型信息熵：")
            calculate_unigram_entropy(tf_dic_train, tf_dic_test)
            print("\n英文二元模型信息熵：")
            calculate_bigram_entropy(tf_dic_train, bigram_tf_dic_train, bigram_tf_dic_test)
            print("\n英文三元模型信息熵：")
            calculate_trigram_entropy(bigram_tf_dic_train, trigram_tf_dic_train, trigram_tf_dic_test)
        except Exception as e:
            print(f"运行时错误：{e}")

if __name__ == '__main__':
    main()