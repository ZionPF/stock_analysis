
# coding: utf-8

# In[1]:


import time
import pandas as pd
from gensim import corpora, models, similarities

import word_segmentation


def load_1h_news():
    """
    加载过去一小时内的所有新闻
    """
    # 1小时时间窗口的历史新闻数据分词结果列表[title+content,],时间戳列表[timestamp]
    all_doc_list = []
    all_timestamp_list = []

    # 数据库通过时间范围查询获取一小时的新闻纪录放入all_doc[timestamp+title+content,]
    # 测试代码，使用../data/news0312.csv作为一小时时间窗口的历史新闻数据
    file_data = pd.read_csv('/data/jupyter/stock/data/news0312.csv')
    ws = word_segmentation.WordSegmentation()
    
    for index, row in file_data.iterrows():
        row_words = ws.word_segmentation(str(row.title) + str(row.content))
        news_ts = row.time
        all_doc_list.append(row_words)
        all_timestamp_list.append(news_ts)
    print("load 1h news success")

    return all_doc_list, all_timestamp_list


class NewsSimilarity(object):

    def __init__(self):
        self.__all_doc_list, self.__all_timestamp_list = load_1h_news()
        self.__dictionary = corpora.Dictionary(self.__all_doc_list)
        self.__corpus = [self.__dictionary.doc2bow(doc) for doc in self.__all_doc_list]
        self.__tfidf = models.TfidfModel(self.__corpus)
        print("tf-idf model has beens build successfully")

    def news_similarity(self, news_seg):
        """
        读取一条新闻的标题和正文，计算该新闻与历史新闻数据的相似度并标记
        :param news_seg: 新闻的标题和正文分词结果（list）
        :return: repeat 是否重复的标记（int）
        """

        # 使用doc2bow将新推送的新闻转换为二元组的向量
        news_vec = self.__dictionary.doc2bow(news_seg)

        # 对corpus语料库中的每个目标文档计算文档的相似度
        index = similarities.SparseMatrixSimilarity(self.__tfidf[self.__corpus], num_features=len(self.__dictionary.keys()))
        sim = index[self.__tfidf[news_vec]]
        print("similarity success")

        # 如果有相似度在90%以上的文档存在，即将新推送的新闻标记为1，否则0
        repeat = 1 if sim.max(axis=0) >= 0.95 else 0

        return repeat

    def add_news(self, news_timestamp, news_seg):
        """
        将新推送的新闻添加至历史数据列表及语料库中
        """
        # 新增推送新闻至all_doc_list（一小时时间窗口历史数据分词结果列表）
        self.__all_timestamp_list.append(news_timestamp)
        self.__all_doc_list.append(news_seg)
        # 新增推送新闻所建立的向量至corpus语料库
        self.__corpus.append(self.__dictionary.doc2bow(news_seg))

    def delete_1h_news(self):
        """
        定时（每小时）删除all_doc_list里前一小时的新闻数据
        """
        time_flag = time.time() - 3600

        index = 0
        for ts in self.__all_timestamp_list:
            if float(ts) >= time_flag:
                del_news = index
                break
            else:
                index += 1

        del self.__all_timestamp_list[0:del_news]
        del self.__all_doc_list[0:del_news]
    
    def tf_idf_model(self):
        """
        重新计算tf-idf模型
        """
        self.__dictionary = corpora.Dictionary(self.__all_doc_list)
        self.__corpus = [self.__dictionary.doc2bow(doc) for doc in self.__all_doc_list]
        self.__tfidf = models.TfidfModel(self.__corpus)
        print("tf-idf model has beens build successfully")


if __name__ == "__main__":

    time_start = time.time()
    # 新推送的新闻
    str_title_content = "孙公司海航地产拟与海南融创昌晟签订《股权转让协议》，出售海航地产所持有的海岛物流100%的股权，转让价款约7.97亿元；同时，海航地产拟出售所持有的海南高和房地产开发有限公司100%的股权至海南融创昌晟，转让价款约11.36亿元。责任编辑：张恒"

    # 新闻分词
    ws = word_segmentation.WordSegmentation()
    news_seg = ws.word_segmentation(str_title_content)
    news_timestamp = '1525682235'
    print("分词结果：")
    print(news_seg)

    # 加载NewsSimilarity类
    news_similarity = NewsSimilarity()

    repeat = news_similarity.news_similarity(news_seg)

    time_elapsed = time.time() - time_start
    print('totally cost {:.0f}s'.format(time_elapsed))

    # 将新推送的新闻添加至历史数据列表及语料库中
    news_similarity.add_news(news_timestamp, news_seg)
    print(repeat)

