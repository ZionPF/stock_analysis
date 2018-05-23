
# coding: utf-8

# In[ ]:

def code_to_stock(str_code_list):
    dict_code = stock_code_dict()
    code_list = str_code_list.split(',')
    stock_list = []
    for code in code_list:
        stock = dict_code[code]
        code_stock = code + '(' + stock + ')'
        stock_list.append(code_stock)
    
    str_stock_list = ','.join(stock_list)
    return str_stock_list

def stock_code_dict():
    # 股票及股票代码表
    data_path = '/data/jupyter/stock/data/stock_list.csv'

    stock_name = []  # 提取出的股票名
    stock_code = []  # 提取出的股票代码

    with open(data_path, 'r', encoding='utf-8') as data:
        for line in data:
            stock_name.append(line[0:-9])
            stock_code.append(line[-8:-2])

    dict_code = dict(zip(stock_code, stock_name))

    return dict_code

if __name__ == '__main__':
    print(code_to_stock('600515'))

