
# coding: utf-8

# # 在线标记资讯关联板块
# 
# * 加载LDA模型
# * 加载RF分类模型
# * 对于资讯，用LDA变为Vector
# * 将Vector进行分类，输出前三的板块名
# 

# In[1]:


# 新闻爬取xlxs文件目录：
NEWS_PATH = '../news/'
#语料库文件路径：
DATA_PATH = '../data/news_words.txt'
#标记文件路径
LABEL_PATH = '../data/news_labels.txt'
#标记数据路径
LABEL_DIR = '../labels/'
# LDA Model 路径：
LDA_PATH = '../model/lda.model'
# Dictionary 路径
DICT_PATH = '../model/dictionary.txtdic'
# Random Forest 模型路径
RF_PATH = '../model/rf.model'

# plate_stock 对应文件
PLATE_STOCK_PATH = '../data/plate_stock.csv'

# stock_plate对应文件
STOCK_PLATE_PATH = '../data/stock_plate.csv'

# 新增新闻标记数据文件
STOCK_TRDATA_PATH = '../labels/stock_training_data.csv'

from sklearn.externals import joblib
#lr是一个LogisticRegression模型

import time
import word_segmentation as ws
from gensim import corpora  
from gensim.models import LdaModel  
from gensim.corpora import Dictionary  
import numpy as np  
import pandas as pd

class PlateLabel():
    def __init__(self,dict_path = DICT_PATH,lda_path = LDA_PATH, rf_path = RF_PATH ):
        self.lda = LdaModel.load(lda_path)
        self.dictionary = corpora.Dictionary.load(dict_path)
        self.rf = joblib.load(rf_path)
        self.plate_stock_data = pd.read_csv(PLATE_STOCK_PATH, dtype=str)
        self.stock_plate_data = pd.read_csv(STOCK_PLATE_PATH, dtype=str)
        
    
    # turn lda result into list
    def lda2list(self,lda,topic_n):
        lda_dict = dict(lda)
        lda_list = [0] * topic_n
        for i in range(topic_n):
            lda_list[i] = lda_dict.get(i,0)
        return lda_list
    
    def plate_forcast(self,news_words_list):
        news_bow = self.dictionary.doc2bow(news_words_list)      #文档转换成bow  
        news_lda = self.lda2list(self.lda[news_bow],301)
        return self.rf.predict([np.array(news_lda)])

    def plate_top3(self,news_words_list,str_code_list):
        in_labels = []
        label_list = []
        news_bow = self.dictionary.doc2bow(news_words_list)      #文档转换成bow  
        news_lda = self.lda2list(self.lda[news_bow],301)
        predict_label = "".join(self.rf.predict([np.array(news_lda)]))
        print(predict_label)
        
        for code in str_code_list:
            code_plate = self.stock_plate_data[self.stock_plate_data.code == code].plate.tolist()
            in_labels += code_plate
        print(in_labels)
        
        if predict_label in in_labels:
            label_list.append(predict_label)
        for i in in_labels:
            if i not in label_list:
                label_list.append(i)
        if predict_label not in label_list:
            label_list = label_list[:2]
            label_list.append(predict_label)
              
        if len(in_labels) == 1:
            with open(STOCK_TRDATA_PATH, 'a', encoding = 'utf-8') as stock_tr_data:
                stock_tr_data.writelines(' '.join(news_words_list) +  ',' + in_labels[0] + '\n')
        if len(label_list) <= 3:
            return label_list
        else:
            return label_list[:3]
#         if len(labels) == 0:
#             label_list += label
#         elif len(labels) <= 2:
#             if label in labels:
#                 label_list += label
#                 for item in labels:
#                     if item != label:
#                         label_list.append(item)
#             else:
#                 for item in labels:
#                     label_list.append(item)
#                 label_list += label
#         else:
#             if label in labels:
#                 label_list += label
#                 for item in labels:
#                     if item != label:
#                         label_list.append(item)
#                 label_list = label_list[0:3]
#             else:
#                 for item in labels:
#                     label_list.append(item)
#                 label_list = label_list[0:2]
#                 label_list += label
                
#         if len(labels) == 1:
#             with open(STOCK_TRDATA_PATH, 'a', encoding = 'utf-8') as stock_tr_data:
#                 stock_tr_data.writelines(' '.join(news_words_list) +  ',' + labels[0] + '\n')
            
#         return label_list
    
    def stock_forcast(self,plate_list):
        stocks = []
        if len(plate_list)>0:
            stocks += self.plate_stock_data[self.plate_stock_data.plate == plate_list[0]].stock.tolist()
        return stocks

if __name__ == "__main__":
    time_start = time.time()
    # 加载新闻分词
    segmentationer = ws.WordSegmentation()
    labeler = PlateLabel()
    #str_title_content = "宝马和戴姆勒牵手背后：行业竞争加剧倒逼的结果 Uber、滴滴等出行平台快速发展，无形中形成一种压力，促使传统汽车巨头走到一起，一方面抢夺出行服务市场的蛋糕，另一方面为自动驾驶技术的推广铺路。近日，德国两大汽车制造商宝马和戴姆勒称将合并各自的汽车共享业务，共同成立一家拥有同等股权的合资企业，其合作领域包括共享汽车、打车服务、停车服务、充电网络、多模式联运等业务。整合后的新公司将成为全球规模最大的共享出行服务商之一，在全球范围内拥有2万辆共享汽车、约14万叫车软件司机和1300万用户以及超过14.3万个充电桩。“我们希望合资公司快速发展，我们已经准备好了大规模的收购。”戴姆勒首席财务官博多·乌博（Bodo Uebber）表示，合资公司的目标是成为一个全球性的参与者。德国贝尔吉施格拉德巴赫应用科学大学汽车管理中心主任Stefan Bratzel在接受外媒采访时表示，戴姆勒与宝马的合作是“朝着完全正确的方向发展，因为移动服务正在向大面积发展。现在不能马上在那里盈利，但有很多基础设施和间接费用可以合并。”“我们合并后的首个目标是要成为行业大玩家，然后才能实现盈利。”宝马董事会成员Peter Schwarzenbauer表示。咨询公司波士顿的一项调查预测显示，到2021年，新车销量将因共享出行减少1%左右，汽车制造商每年遭受的损失将超过80亿美元。这意味着，为了能够实现持续性的盈利，这些汽车制造商必然要考虑出行服务市场。实际上，戴姆勒和宝马都在较早布局共享汽车业务的传统车企。汽车共享项目“car2go”成立于2008年，宝马的“Drive Now”于2011年推出。不过，当前由于运营成本较高等因素目前两者均未实现盈利。合并业务一方面能够共享运营资源从而降低成本，另一方面可以给用户提供更多的车辆选择和捆绑优惠。不过，与互联网企业相比，传统车企并不具备运营出行服务平台的先天优势。“出行服务平台的建设涉及到大数据、IT技术以及云平台等方面的建设，汽车企业在这方面有所欠缺。但最近几年，他们也在收购一些科技公司来弥补互联网技术的缺失。此外，互联网企业公司的文化是比较灵活的，反应迅速。”汽车行业专家颜景辉对记者表示，传统汽车厂商也在加快转变，比如奥迪就在最近举办的2017年财报年会上称要在组织结构上进行变革，以达到更加高效的工作效率，其成立的创新中心已经采用了互联网企业的工作方式。毕马威企业咨询(中国）有限公司合伙人Huu-Hoi Tran在接受第一财经记者采访时表示，宝马和戴姆勒走向合作就是竞争压力造成的结果。“但现在还无法判断未来究竟谁能够取胜，这要看用户的体验和忠诚度。”在商务部研究院国际市场研究所副所长白明看来，宝马和戴姆勒逐渐从汽车制造商延伸业务范围，从制造到使用扩大了经营范围。提高共享出行的使用环境，有助于在自动驾驶汽车等很多方面获得更高的起点。“共享汽车让我们更快引入自动驾驶技术，同时影响共享汽车业务的成本架构。”通用汽车战略与全球业务规划副总裁迈克尔·艾博森曾在第七届全球汽车论坛时公开表示。而目前没有任何一家公司可以做到对自动驾驶技术百分百的放心，在这样的情况下，如果将自动驾驶技术直接搭载在量产车型上，那么将由普通消费者来承担与安全相关的风险。如果在共享出行的车辆上应用自动驾驶技术，风险就由服务提供商来承担，这让自动驾驶技术的推广在安全方面降低了很多风险。科技巨头也在加紧在自动驾驶领域的争夺。近日，谷歌旗下的无人驾驶部门Waymo宣布，已与印度塔塔汽车旗下的英国汽车制造商捷豹路虎达成为期八年的合作协议（2026年到期），计划购入2万辆捷豹I-Pace纯电动化SUV，作为其今年晚些时候推出的自动驾驶打车服务主力车型，而滴滴出行也正在加快在自动驾驶领域的研发。这意味着，如果软件技术缺乏的传统汽车厂商仍然只限于生产制造端的话，未来就可能沦为这些科技巨头的代工厂。汽车厂商自然不想沦为代工厂，比如，在百度做自动驾驶初期，宝马曾与其合作，但仅维持了两个月左右。宝马中国CEO康思远(Olaf Kastner)曾为此解释称，两个公司的发展步伐以及经营理念有一些不同，对今后如何进行研究存在分歧。从今天的百度阿波罗计划看来，百度是想要开展自己握有主动权的自动驾驶平台，而宝马自然不想被夺取主动权。“总有一天，出行平台和硬件制造商之间会有一场战争。”滴滴出行CEO程维曾公开表示。宝马和戴姆勒合并共享汽车业务，意味传统车企巨头将向出行平台宣战，在汽车产业革命的浪潮中，究竟谁能够成为最大的玩家，这拭目以待。"
    str1 = "海航基础（600515）3月12日晚间公告，孙公司海航地产拟与海南融创昌晟签订《股权转让协议》，出售海航地产所持有的海岛物流100%的股权，转让价款约7.97亿元；同时，海航地产拟出售所持有的海南高和房地产开发有限公司100%的股权至海南融创昌晟，转让价款约11.36亿元。责任编辑：张恒"
    # 新闻分词
    segmentationer = ws.WordSegmentation()
    news_seg = segmentationer.word_segmentation(str1)
    print('Initiating cost {:.0f}s'.format(time.time()-time_start))
    time_start = time.time()
    #label =  labeler.plate_forcast(news_seg)
    str_code_list = ['300513']
    labels = labeler.plate_top3(news_seg, str_code_list)
    print(labels)
    stocks = labeler.stock_forcast(labels)
    print(stocks)
    time_elapsed = time.time() - time_start
    
    print('totally cost {:.0f}s'.format(time.time()-time_start))

