
# coding: utf-8

# In[1]:


import json
from flask import Flask
from flask import request
import word_segmentation as ws
import news_similarity as ns
import extract_stock_code as esc
import news_plate_label as pl

WordSegmentation = ws.WordSegmentation()
NewsSimilarity = ns.NewsSimilarity()
ExtractStockCode = esc.ExtractStockCode()
NewsPlateLabel = pl.PlateLabel()

app = Flask(__name__)
Counter = 0

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/data/v1.0/getLabels/', methods=['POST'])
def getLabels():
    global Counter
#     print(request.form['title'])
#     print(request.form['content'])
    str_time = request.form['time']
    str_title = request.form['title']
    str_content = request.form['content']
    print("-------------")
    print("time:",str_time)
    print("title:",str_title)
    news_seg = WordSegmentation.word_segmentation(str_title + str_content)
#     print(news_seg)
    str_code_list = ExtractStockCode.extract_stock_code(news_seg)
    print("include stock:",str_code_list)
    repeat = NewsSimilarity.news_similarity(news_seg)
    print("If repeat:",repeat)
    NewsSimilarity.add_news(str_time, news_seg)
    code_list = str_code_list.split(',')
    Counter += 1
#     print(Counter)
    
    if Counter == 100:
        NewsSimilarity.delete_1h_news()
        NewsSimilarity.tf_idf_model()
        Counter = 0
    
    #plate_label =  NewsPlateLabel.plate_forcast(news_seg)
    plate_label = NewsPlateLabel.plate_top3(news_seg, code_list)
    plate_string = ",".join(plate_label)
    print("forcasted plates:",plate_label)
    
    stock_forcast = NewsPlateLabel.stock_forcast(plate_label)
    stock_forcast_str = ",".join([str(x) for x in stock_forcast])
    print("forcasted stocks:",stock_forcast)
    dic_output = dict(stock_code = str_code_list, repeat = repeat, plate = plate_string, stock_forcast = stock_forcast_str)
    json_output = json.dumps(dic_output)
    return json_output

if __name__ == '__main__':
    app.run("0.0.0.0")

