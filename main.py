from nl2tensor import *
from process_control import *
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from language_model.transformers import ElectraTokenizer
from nn.embeddings import ElectraModel
from language_model.transformers.configuration_electra import ElectraConfig
from selenium import webdriver
from bs4 import BeautifulSoup
from data.data_get import read_lines
from data.data_process import bio2xml
import urllib.request
import torch
import os
import re
import glob
import argparse
import json
from text_classify_train import TextClassifyModel
from sent_ner_train import NerModel


def set_args(filename):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_classify_model_save_dir",
                        default='checkpoint/text_classify/',
                        type=str)
    parser.add_argument("--text_classify_model_config_dir",
                        default='config/text_classify_config.json',
                        type=str)
    parser.add_argument("--sent_ner_model_save_dir",
                        default='checkpoint/sent_ner/',
                        type=str)
    parser.add_argument("--sent_ner_model_config_dir",
                        default='config/sent_ner_config.json',
                        type=str)
    parser.add_argument("--data_dir",
                        default='data/cut_data/',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--driver_dir",
                        default='data/chrome_driver.exe',
                        type=str)
    parser.add_argument("--vocab_dir",
                        default='pretrained_model/pytorch_electra_180g_large/vocab.txt',
                        type=str,
                        help="The vocab data dir.")
    parser.add_argument("--max_sent_len",
                        default=128,
                        type=int,
                        help="句子最大字符数")
    parser.add_argument("--max_text_len",
                        default=256,
                        type=int,
                        help="文本最大长度")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="验证模式")
    parser.add_argument("--no_gpu",
                        default=False,
                        action='store_true',
                        help="用不用gpu")
    parser.add_argument("--fp16",
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true')
    parser.add_argument("--text2label",
                        default={'交通事故': 0, '地震': 1, '恐怖袭击': 2, '火灾': 3, '食物中毒': 4},
                        type=dict)
    parser.add_argument("--label2text",
                        default={0: '交通事故', 1: '地震', 2: '恐怖袭击', 3: '火灾', 4: '食物中毒'},
                        type=dict)
    parser.add_argument("--START_TAG",
                        default="[CLS]",
                        type=str,
                        help="序列开始标识符")
    parser.add_argument("--STOP_TAG",
                        default="[SEP]",
                        type=str,
                        help="序列停止标识符")
    parser.add_argument("--MASK_TAG",
                        default="[MASK]",
                        type=str,
                        help="序列掩码符号")
    parser.add_argument("--tag_to_ix",
                        default={"B-Time": 0, "I-Time": 1, "B-Location": 2, "I-Location": 3, "B-Object": 4,
                                 "I-Object": 5, "B-Participant": 6, "I-Participant": 7, "B-Means": 8, "I-Means": 9,
                                 "B-Denoter": 10, "I-Denoter": 11, "o": 12, '[CLS]': 13, "[SEP]": 14, "[MASK]": 15},
                        type=list)
    parser.add_argument("--ix_to_tag",
                        default={0: "B-Time", 1: "I-Time", 2: "B-Location", 3: "I-Location", 4: "B-Object",
                                 5: "I-Object", 6: "B-Participant", 7: "I-Participant", 8: "B-Means", 9: "I-Means",
                                 10: "B-Denoter", 11: "I-Denoter", 12: "o", 13: '[CLS]', 14: "[SEP]", 15: "[MASK]"},
                        type=list)
    parser.add_argument("--tag_to_tag",
                        default={"B-Time": "Time", "I-Time": "Time", "B-Location": "Location",
                                 "I-Location": "Location", "B-Object": "Object", "I-Object": "Object",
                                 "B-Participant": "Participant", "I-Participant": "Participant",
                                 "B-Means": "Means", "I-Means": "Means", "B-Denoter": "Denoter",
                                 "I-Denoter": "Denoter", "o": "Other", '[CLS]': "Other", "[SEP]": "Other",
                                 "[MASK]": "Other"},
                        type=list)
    args = parser.parse_args()
    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args


class MyTextClassifyModel:
    def __init__(self, text_classify_model):
        self.TextClassifyModel = text_classify_model

    # 根据输出预测label
    def label_from_output(self, output):
        _, top_i = output.data.topk(1)
        return top_i[0]

    def get_label(self, input):
        guess = self.TextClassifyModel.get_guess(input)
        return args.label2text[self.label_from_output(guess).item()]


class MyNerModel:
    def __init__(self, ner_model):
        self.NerModel = ner_model

    # 根据输出预测label
    def label_from_output(self, output):
        _, top_i = output.data.topk(1)
        return top_i[0]

    def get_bio(self, input, tokened_sent):
        tags = []
        tokens = []
        tag_seq = self.NerModel.get_tag_seq(input)[0]
        for idx, token in enumerate(tokened_sent):
            if token != args.START_TAG and token != args.STOP_TAG and token != args.MASK_TAG:
                try:
                    tokens.append(token)
                    tags.append(args.tag_to_tag[args.ix_to_tag[self.label_from_output(tag_seq[idx]).item()]])
                except IndexError:
                    continue
        return tags, tokens


class MyTextClassify:
    def __init__(self, language_model, my_text_classify_model, tokenizer):
        self.LanguageModel = language_model
        self.MyTextClassifyModel = my_text_classify_model
        self.tokenizer = tokenizer

    def get_label(self, text):
        with torch.no_grad():
            b_input_ids, b_input_mask, _ = text2ids(self.tokenizer, text, args.max_text_len)
            b_input_ids = torch.Tensor(b_input_ids).to(device)
            b_input_mask = torch.Tensor(b_input_mask).to(device)
            text_embedding = self.LanguageModel(input_ids=b_input_ids.squeeze(1).long(),
                                                attention_mask=b_input_mask)
            return self.MyTextClassifyModel.get_label(text_embedding)


class MySentNer:
    def __init__(self, language_model, my_sent_ner_model, tokenizer):
        self.LanguageModel = language_model
        self.MyNerModel = my_sent_ner_model
        self.tokenizer = tokenizer

    # tokens切分文本
    def text2tokens(self, text):
        tokens_a = self.tokenizer.tokenize(text)
        tokens = []
        tokens.append(args.START_TAG)
        for token in tokens_a:
            tokens.append(token)
        tokens.append(args.STOP_TAG)
        tokens.append(args.STOP_TAG)
        return tokens

    def get_bio(self, sent):
        with torch.no_grad():
            b_input_ids, b_input_mask, _ = text2ids(self.tokenizer, sent, args.max_sent_len)
            b_input_ids = torch.Tensor(b_input_ids).to(device)
            b_input_mask = torch.Tensor(b_input_mask).to(device)
            sent_embedding = self.LanguageModel(input_ids=b_input_ids.squeeze(1).long(),
                                                attention_mask=b_input_mask)
            tokened_sent = self.text2tokens(sent)
            return self.MyNerModel.get_bio(sent_embedding, tokened_sent)


class WebData:
    def __init__(self, my_sent_ner, my_text_classify):
        self.SentNer = my_sent_ner
        self.TextClassify = my_text_classify

    def cut_sent(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

    def test_sel(self, keyword, link):
        # Chrome驱动器的地址，可以换成Edge,Firefox或Ie,驱动器地址网上搜。需要安装版本数接近的Chrome浏览器
        driver = webdriver.Chrome(executable_path=args.driver_dir)
        driver.get(link)
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.ID, "ww"))
            )
        except TimeoutException:
            print('加载页面失败')
        try:
            element = driver.find_element_by_css_selector('#ww')
            print('成功找到了搜索框')
            keyword = keyword
            print(keyword)
            print('输入关键字', keyword)
            element.send_keys(keyword)
            element.send_keys(Keys.ENTER)
        except NoSuchElementException:
            print('没有找到搜索框')
        print(u'正在查询该关键字')
        html = driver.page_source
        num1 = self.crawl_html(html, 0)
        is_true = True
        while is_true:
            try:
                driver.find_element_by_link_text('下一页>').click()
                html = driver.page_source
                num1 = self._crawl_html(html, num1)
                if num1 > 50:  # 查找次数
                    is_true = False
            except NoSuchElementException:
                print('没有点击按钮')
                is_true = False
        print('完成啦！')
        return

    def crawl_html(self, html,  num):  # 根据需要自己改
        bf = BeautifulSoup(html, 'lxml')
        for i in range(0, 20):
            all_texts = bf.find_all('h3', recursive=True)
            try:
                texts = all_texts[i].find('a')
            except IndexError:
                break
            url_news = texts['href']
            page_req_news = urllib.request.urlopen(url_news).read()
            page_news = page_req_news
            bf_news = BeautifulSoup(page_news, 'lxml')
            bf_news_title = bf_news.find('head').find('title')
            if bf_news_title is not None:
                bf_news_title = bf_news_title.text
                num += 1
                text = ""
                para = ""
                bios, tokend_text = [], []
                bf_news_contents = bf_news.find('body')
                contents = bf_news_contents.find_all(class_='bjh-p')
                if len(contents) == 0:
                    contents = bf_news_contents.find_all(class_='cont')
                for content in contents:
                    content = str(re.sub(u"<.*>", "", content.text))
                    if content[-1] not in [',', '.', '?', '!', "”", "'", "\"", "，", "。", "？", "！"]:
                        text = text + content + '。'
                    else:
                        text += content
                for line in self.cut_sent(text):
                    para += line
                    bio, tokend_sent = self.SentNer.get_bio(line)
                    bios.append(bio)
                    tokend_text.append(tokend_sent)
                label = self.TextClassify.get_label(para)
                self.mkdir_path("web_data//" + label)
                try:
                    with open("web_data//%s//%s.txt" % (label, bf_news_title), "w", encoding='utf-8') as f:
                        for bio, tokend_sent in zip(bios, tokend_text):
                            for tag, token in zip(bio, tokend_sent):
                                f.writelines("{}\t{}".format(token, tag))
                                f.writelines("\n")
                            f.writelines("\n")
                        f.close()
                except OSError:
                    with open("web_data//%s//%s.txt" % (label, str(num)+'-'+str(i)), "w", encoding='utf-8') as f:
                        for bio, tokend_sent in zip(bios, tokend_text):
                            for tag, token in zip(bio, tokend_sent):
                                f.writelines("{}\t{}".format(token, tag))
                                f.writelines("\n")
                            f.writelines("\n")
                        f.close()
        return num

    def mkdir_path(self, path):
        path = path.strip()  # 去除首位空格
        path = path.rstrip("\\")  # 去除尾部 \ 符号
        is_exists = os.path.exists(path)  # 判断路径是否存在
        if not is_exists:  # 判断结果
            os.makedirs(path)  # 如果不存在则创建目录
            print(path + ' 创建成功')
            return True
        else:
            return False

    def get_web_data(self, link='http://news.baidu.com/?tn=news'):
        keys = []
        print('输入搜索关键词，输入0结束')
        while True:
            key_word = input()
            if key_word != '0':
                keys.append(key_word)
            else:
                print('一股来自东方的神秘力量正在试图掌控你的浏览器...')
                break
        for key in keys:
            self.test_sel(key, link)


if __name__ == "__main__":
    # 初始化，加载环境
    try:
        args = set_args('config/event_extraction_args.txt')
    except FileNotFoundError:
        args = set_args('config/event_extraction_args.txt')
    logger = get_logger()
    set_environ()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    tokenizer = ElectraTokenizer.from_pretrained(args.vocab_dir)

    # 初始化网络
    text_classify_config = ElectraConfig.from_pretrained(args.text_classify_model_config_dir)
    text_classify_embedding = ElectraModel(config=text_classify_config)
    text_classify_model = TextClassifyModel(config=text_classify_config)
    output_model_file = os.path.join(args.text_classify_model_save_dir, 'embedding/')
    model_state_dict = torch.load(os.path.join(output_model_file, 'pytorch_model.bin'))
    text_classify_embedding.load_state_dict(model_state_dict)
    output_model_file = os.path.join(args.text_classify_model_save_dir, "mymodel.bin")
    model_state_dict = torch.load(output_model_file)
    text_classify_model.load_state_dict(model_state_dict)

    sent_ner_config = ElectraConfig.from_pretrained(args.sent_ner_model_config_dir)
    sent_ner_embedding = ElectraModel(config=sent_ner_config)
    sent_ner_model = NerModel(config=sent_ner_config)
    output_model_file = os.path.join(args.sent_ner_model_save_dir, 'embedding/')
    model_state_dict = torch.load(os.path.join(output_model_file, 'pytorch_model.bin'))
    sent_ner_embedding.load_state_dict(model_state_dict)
    output_model_file = os.path.join(args.sent_ner_model_save_dir, "mymodel.bin")
    model_state_dict = torch.load(output_model_file)
    sent_ner_model.load_state_dict(model_state_dict)

    text_classify_embedding.to(device)
    text_classify_model.to(device)
    sent_ner_embedding.to(device)
    sent_ner_model.to(device)

    text_classify_embedding.eval()
    text_classify_model.eval()
    sent_ner_embedding.eval()
    sent_ner_model.eval()

    text_classify_model = MyTextClassifyModel(text_classify_model)
    sent_ner_model = MyNerModel(sent_ner_model)
    text_classify = MyTextClassify(text_classify_embedding, text_classify_model, tokenizer)
    sent_ner = MySentNer(sent_ner_embedding, sent_ner_model, tokenizer)

    # 功能选择
    while True:
        choice = input("请选择功能，输入0退出;" '\n'
                       "输入1从网络抽取信息;" '\n'
                       "输入2从语料库抽取信息;" '\n'
                       "输入3将bio标注数据集转化成xml格式;" '\n'
                       "输入4将重新训练事件分类网络;" '\n'
                       "输入5将重新训练事件要素识别网络;" '\n'
                       "输入6后，输入一段文字，将自动识别出其中的要素。" '\n')
        if choice == "0":
            break
        if choice == "1":
            # 从网络抽取数据
            web_data = WebData(my_sent_ner=sent_ner, my_text_classify=text_classify)
            web_data.get_web_data()
            continue
        if choice == "2":
            # 从数据库抽取数据
            Name = re.compile('\\\\.*\.txt')
            print("正在抽取数据")
            all_dir = glob.glob(args.data_dir+'*')
            for dir_name in all_dir:
                all_filenames = glob.glob(dir_name + "/*.txt")
                for idx, filename in enumerate(all_filenames):
                    text = ""
                    bios = []
                    tokend_text = []
                    for line in read_lines(filename):
                        text += line
                        bio, tokend_sent = sent_ner.get_bio(line)
                        bios.append(bio)
                        tokend_text.append(tokend_sent)
                    label = text_classify.get_label(text)
                    name = re.sub(u"\\\\.*?\\\\", "", Name.findall(filename)[0])
                    f = open("base_result/"+label+"/"+name, 'w', encoding='utf-8')
                    for bio, tokend_sent in zip(bios, tokend_text):
                        for tag, token in zip(bio, tokend_sent):
                            f.writelines("{}\t{}" .format(token, tag))
                            f.writelines("\n")
                        f.writelines("\n")
                    f.close()
            print("已完成事件抽取")
            continue
        if choice == '3':
            data_paths = glob.glob("web_data/*")
            result_paths = glob.glob("xml_result/*")
            for data_path, result_path in zip(data_paths, result_paths):
                bio2xml(data_path, result_path, args.tag_to_tag)
            print("转化完成!")
            continue
        if choice == '4':
            os.system('text_classify_train.py')
            continue
        if choice == '5':
            os.system('sent_ner_train.py')
            continue
        if choice == '6':
            line = input("请输入一句话，输入完成后按回车键"'\n')
            label = text_classify.get_label(line)
            print("类别:{}".format(label))
            bios, words = sent_ner.get_bio(line)
            last_bio = bios[0]
            if last_bio != 'Other':
                print("{}:".format(last_bio), end='')
            if not (np.array(bios) == 'Other').all():
                for word, bio in zip(words, bios):
                    if bio != 'Other' and bio != last_bio and last_bio != 'Other':
                        last_bio = bio
                        print('\n')
                        print("{}:{}".format(bio, word), end='')
                    elif bio != 'Other' and last_bio == 'Other':
                        last_bio = bio
                        print('\n')
                        print("{}:{}".format(bio, word), end='')
                    elif bio != 'Other' and bio == last_bio:
                        print(word, end='')
                    elif bio == 'Other' and bio != last_bio:
                        last_bio = bio
                        print(word, end='')
                    elif bio == 'Other' and bio == last_bio:
                        print(word, end='')
                print('\n')
            else:
                print(line)
            continue
        else:
            print("输入错误")
            continue
