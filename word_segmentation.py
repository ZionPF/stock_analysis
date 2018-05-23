
# coding: utf-8

# In[ ]:


import jieba
import unicodedata
import utils


def stop_words_list():
    # 停用词文件
    data_path = '/data/jupyter/stock/utils/stopwords.txt'

    with open(data_path, 'r', encoding='utf-8') as data:
        stopwords = [line.strip() for line in data.readlines()]

    temp_stop_list = ['\u3000', '\xa0', '\t']
    stop_words = stopwords + temp_stop_list
    return stop_words

def is_number(s):
    # 判断是否为数字
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


class WordSegmentation(object):

    def __init__(self):
        self.dict_code = utils.stock_code_dict()
        self.stopword_list = stop_words_list()

    def word_segmentation(self, str_title_content):

        # 结巴分词词库加载股票名词
        jieba.load_userdict('/data/jupyter/stock/data/user_dict.txt')

        # 分词结果列表
        news_list = []

        str_content = str(str_title_content).replace('\t', '').replace('\n', '').replace('\r', '').replace(' ', '')
        str_words = ','.join(jieba.cut_for_search(str_content)).split(',')

        for word in str_words:
            if word not in self.stopword_list:
                if word[-1] != '%':
                    if is_number(word):
                        if word in self.dict_code:
                            news_list.append(word)
                    else:
                        news_list.append(word)

        return news_list

if __name__ == "__main__":
    str_title_content = "新浪财经讯 3月12日消息，海航基础（600515）3月12日晚间公告，孙公司海航地产拟与海南融创昌晟签订《股权转让协议》，出售海航地产所持有的海岛物流100%的股权，转让价款约7.97亿元；同时，海航地产拟出售所持有的海南高和房地产开发有限公司100%的股权至海南融创昌晟，转让价款约11.36亿元。责任编辑：张恒"
    
    # 加载分词类
    ws = WordSegmentation()
    news_seg = ws.word_segmentation(str_title_content)
    print("分词结果：")
    print(news_seg)
    
    
