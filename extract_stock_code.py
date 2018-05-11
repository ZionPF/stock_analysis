
# coding: utf-8

# In[1]:


import time

import word_segmentation

def load_stock_code():
    """
      加载数据库里的股票_代码字典
      dict_stock({股票名：股票代码})
      dict_code({股票代码：股票名})
    """

    data_path = '/data/jupyter/stock/data/stock_list.csv'

    stock_name = []  # 提取出的股票名
    stock_code = []  # 提取出的股票代码

    with open(data_path, 'r', encoding='utf-8') as data:
        for line in data:
            stock_name.append(line[0:-9])
            stock_code.append(line[-8:-2])

    dict_stock = dict(zip(stock_name, stock_code))
    dict_code = dict(zip(stock_code, stock_name))

    return dict_stock, dict_code

class ExtractStockCode(object):

    def __init__(self):
        self.__dict_stock, self.__dict_code = load_stock_code()

    def extract_stock_code(self, news_list):
        """
        读取一条新闻的标题和正文，提取其中出现的股票名和股票代码，将股票名转换成股票代码，最后返回所有的股票代码
        :param: 新闻的标题和正文分词结果（list）
        :returns：包含的所有股票代码（string：code1,code2...）
        """

        list_code = []  # 提取的股票代码列表

        for word in news_list:
            if word in self.__dict_stock:
                word = self.__dict_stock[word]
                if word not in list_code:
                    list_code.append(word)

            if (word in self.__dict_code and word not in list_code):
                list_code.append(word)
                
        str_code_list = ','.join(list_code)

        return str_code_list


# Extract_stocke_code test
if __name__ == "__main__":
    # 新推送的新闻
    str_title_content = "新浪财经讯 3月12日消息，海航基础（600515）3月12日晚间公告，孙公司海航地产拟与海南融创昌晟签订《股权转让协议》，出售海航地产所持有的海岛物流100%的股权，转让价款约7.97亿元；同时，海航地产拟出售所持有的海南高和房地产开发有限公司100%的股权至海南融创昌晟，转让价款约11.36亿元。责任编辑：张恒"

    # 新闻分词
    ws = word_segmentation.WordSegmentation()
    news_list = ws.word_segmentation(str_title_content)

    print("分词结果：")
    print(news_list)

    time_start = time.time()

    # 加载ExtractStockCode类
    extract_stock_code = ExtractStockCode()

    # 提取股票代码
    str_code_list = extract_stock_code.extract_stock_code(news_list)

    time_elapsed = time.time() - time_start
    print('totally cost {:.0f}ms'.format(time_elapsed * 1000))

    print("股票代码：")
    print(str_code_list)

