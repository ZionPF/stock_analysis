
# coding: utf-8

# # 在线标记资讯关联板块
# 
# * 加载LDA模型
# * 加载RF分类模型
# * 对于资讯，用LDA变为Vector
# * 将Vector进行分类，输出前三的板块名
# 

# In[3]:


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

    def plate_top3(self,news_words_list):
        news_bow = self.dictionary.doc2bow(news_words_list)      #文档转换成bow  
        news_lda = self.lda2list(self.lda[news_bow],301)
        return self.rf.predict_proba([np.array(news_lda)])
    
    def stock_forcast(self,plate_list):
        stocks = []
        for i in plate_list:
            stocks += self.plate_stock_data[self.plate_stock_data.plate == i].stock.tolist()
        return stocks


# In[4]:


if __name__ == "__main__":
    time_start = time.time()
    # 加载新闻分词
    segmentationer = ws.WordSegmentation()
    labeler = PlateLabel()
    str_title_content = "宝马和戴姆勒牵手背后：行业竞争加剧倒逼的结果 Uber、滴滴等出行平台快速发展，无形中形成一种压力，促使传统汽车巨头走到一起，一方面抢夺出行服务市场的蛋糕，另一方面为自动驾驶技术的推广铺路。近日，德国两大汽车制造商宝马和戴姆勒称将合并各自的汽车共享业务，共同成立一家拥有同等股权的合资企业，其合作领域包括共享汽车、打车服务、停车服务、充电网络、多模式联运等业务。整合后的新公司将成为全球规模最大的共享出行服务商之一，在全球范围内拥有2万辆共享汽车、约14万叫车软件司机和1300万用户以及超过14.3万个充电桩。“我们希望合资公司快速发展，我们已经准备好了大规模的收购。”戴姆勒首席财务官博多·乌博（Bodo Uebber）表示，合资公司的目标是成为一个全球性的参与者。德国贝尔吉施格拉德巴赫应用科学大学汽车管理中心主任Stefan Bratzel在接受外媒采访时表示，戴姆勒与宝马的合作是“朝着完全正确的方向发展，因为移动服务正在向大面积发展。现在不能马上在那里盈利，但有很多基础设施和间接费用可以合并。”“我们合并后的首个目标是要成为行业大玩家，然后才能实现盈利。”宝马董事会成员Peter Schwarzenbauer表示。咨询公司波士顿的一项调查预测显示，到2021年，新车销量将因共享出行减少1%左右，汽车制造商每年遭受的损失将超过80亿美元。这意味着，为了能够实现持续性的盈利，这些汽车制造商必然要考虑出行服务市场。实际上，戴姆勒和宝马都在较早布局共享汽车业务的传统车企。汽车共享项目“car2go”成立于2008年，宝马的“Drive Now”于2011年推出。不过，当前由于运营成本较高等因素目前两者均未实现盈利。合并业务一方面能够共享运营资源从而降低成本，另一方面可以给用户提供更多的车辆选择和捆绑优惠。不过，与互联网企业相比，传统车企并不具备运营出行服务平台的先天优势。“出行服务平台的建设涉及到大数据、IT技术以及云平台等方面的建设，汽车企业在这方面有所欠缺。但最近几年，他们也在收购一些科技公司来弥补互联网技术的缺失。此外，互联网企业公司的文化是比较灵活的，反应迅速。”汽车行业专家颜景辉对记者表示，传统汽车厂商也在加快转变，比如奥迪就在最近举办的2017年财报年会上称要在组织结构上进行变革，以达到更加高效的工作效率，其成立的创新中心已经采用了互联网企业的工作方式。毕马威企业咨询(中国）有限公司合伙人Huu-Hoi Tran在接受第一财经记者采访时表示，宝马和戴姆勒走向合作就是竞争压力造成的结果。“但现在还无法判断未来究竟谁能够取胜，这要看用户的体验和忠诚度。”在商务部研究院国际市场研究所副所长白明看来，宝马和戴姆勒逐渐从汽车制造商延伸业务范围，从制造到使用扩大了经营范围。提高共享出行的使用环境，有助于在自动驾驶汽车等很多方面获得更高的起点。“共享汽车让我们更快引入自动驾驶技术，同时影响共享汽车业务的成本架构。”通用汽车战略与全球业务规划副总裁迈克尔·艾博森曾在第七届全球汽车论坛时公开表示。而目前没有任何一家公司可以做到对自动驾驶技术百分百的放心，在这样的情况下，如果将自动驾驶技术直接搭载在量产车型上，那么将由普通消费者来承担与安全相关的风险。如果在共享出行的车辆上应用自动驾驶技术，风险就由服务提供商来承担，这让自动驾驶技术的推广在安全方面降低了很多风险。科技巨头也在加紧在自动驾驶领域的争夺。近日，谷歌旗下的无人驾驶部门Waymo宣布，已与印度塔塔汽车旗下的英国汽车制造商捷豹路虎达成为期八年的合作协议（2026年到期），计划购入2万辆捷豹I-Pace纯电动化SUV，作为其今年晚些时候推出的自动驾驶打车服务主力车型，而滴滴出行也正在加快在自动驾驶领域的研发。这意味着，如果软件技术缺乏的传统汽车厂商仍然只限于生产制造端的话，未来就可能沦为这些科技巨头的代工厂。汽车厂商自然不想沦为代工厂，比如，在百度做自动驾驶初期，宝马曾与其合作，但仅维持了两个月左右。宝马中国CEO康思远(Olaf Kastner)曾为此解释称，两个公司的发展步伐以及经营理念有一些不同，对今后如何进行研究存在分歧。从今天的百度阿波罗计划看来，百度是想要开展自己握有主动权的自动驾驶平台，而宝马自然不想被夺取主动权。“总有一天，出行平台和硬件制造商之间会有一场战争。”滴滴出行CEO程维曾公开表示。宝马和戴姆勒合并共享汽车业务，意味传统车企巨头将向出行平台宣战，在汽车产业革命的浪潮中，究竟谁能够成为最大的玩家，这拭目以待。"
    str1 = "周一早盘沪深两市股指高开震荡，受银行股强势走高的带动，上证50指数盘初快速冲高，涨幅一度逼近1%，沪指也一度逼近3450点关口。深圳市场上，由于新能源及芯片概念再度趋于活跃，中小板指盘中震荡走高，量能释放显著。截至上午收盘，沪深两市主要股指全面飘红，中小板指涨幅达到0.75%，两市成交量较上一交易日显著放量。午市收盘数据（来源：Wind）沪深两市上午收盘，上证综指收报3,442.99点,上涨10.32点，涨幅0.3%，成交额1,534亿元；深证成指收报11,692.46点,上涨47.41点，涨幅0.41%，成交额1,877亿元；创业板指收报1,906.92点,上涨6.29点，涨幅0.33%，成交额506亿元。资金方面，央行今日进行800亿元7天逆回购操作、700亿元14天期逆回购操作、300亿元63天期逆回购操作，今日有300亿元逆回购到期。此外今日还有665亿MLF到期。盘面上，银行、芯片、参股360、太阳能以及人脸识别等概念板块涨幅居前。跌幅榜上，小程序、包装印刷、物流、农机以及造纸板块领跌。热点板块：银行板块周一早盘强势领涨，其中次新银行股涨幅居前。截至中午收盘时，吴江银行涨停，平安银行、无锡银行涨逾5%，常熟银行、江阴银行、张家港行涨逾4%，招商银行、贵阳银行涨逾3%。消息面上，11月10日，财政部副部长朱光耀在国新办吹风会上表示，中方将对外资参股境内金融机构放宽持股比例限制，部分金融机构将允许外资控股。华泰证券表示，财政部拟放松外资对银行的持股比例上限，为战略股东继续增持银行股打开空间，将刺激银行股抢筹行情。风电、光伏等相关领域龙头早盘纷纷走强，带动新能源板块整体上扬。截至中午收盘，南玻A涨停，林洋能源、扬杰科技、北方华创涨逾8%，天顺风能、湘电股份涨逾7%，阳光电源、金风科技涨逾6%，金辰股份涨逾5%。跌幅榜上，小程序、包装印刷、物流、农机以及造纸板块领跌。消息面：1、财政部、税务总局、证监会联合发布通知，对内地个人投资者通过沪港通投资香港联交所上市股票取得的转让差价所得，自2017年11月17日起至2019年12月4日止，继续暂免征收个人所得税。相关负责人表示，对内地个人买卖港股差价三年内暂免征收个人所得税属于此次沪港通税收政策的优惠措施之一。主要考虑沪港通尚处于启动试点阶段，为吸引内地个人赴港投资，推动沪港通的开展，在税收政策方面给予适当优惠。2、体育总局网站消息称，要动员社会各方面力量全力做好2022年北京冬奥会、冬残奥会各项筹办工作。要加快体育产业发展，使体育产业成为体育事业发展的助推器和经济转型升级的新动能。要进一步深化体育改革，破解制约体育强国建设的体制机制障碍。3、印尼邦加勿里洞省长Erzaldo Rosman称，正在起草锡衍生矿物，即禁止稀土金属出口的省长条例，该省长条例将详述投资和加工含率，直至能进行贸易。机构观点：平安证券预计，境外资金将持续加速流入A股市场，成为A股市场不容小觑的做多势力，“慢牛”行情仍将延续。在此影响下，境外资金在A股市场定价上扮演着愈发重要的角色，结构上的龙头估值溢价将持续。渤海证券指出，考虑到当前业绩与估值一正一反的两相作用，行情单边发展的可能性被削弱了，市场将大概率延续年初以来的低波动特征。而风格上，由于小盘股年报中的商誉减计风险仍待释放，以及估值的难言便宜，板块短期内尚难具有反转预期。而中大市值板块虽然涨幅已经较深，且部分预期拥挤个股的估值偏贵，但仍难掩其动量仍在，因而风格上延续大小估值弥合过程的概率较大。"
    # 新闻分词
    segmentationer = ws.WordSegmentation()
    news_seg = segmentationer.word_segmentation(str1)
    print('Initiating cost {:.0f}s'.format(time.time()-time_start))
    time_start = time.time()
    label =  labeler.plate_forcast(news_seg)
    print(label)
    stocks = labeler.stock_forcast(label)
    print(stocks)
    time_elapsed = time.time() - time_start
    
    print('totally cost {:.0f}s'.format(time.time()-time_start))

