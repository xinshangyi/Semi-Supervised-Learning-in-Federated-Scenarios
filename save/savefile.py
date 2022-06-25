import pandas as pd

def savexcl(savelist, filename):
	# 列表
    # list转dataframe
    df = pd.DataFrame(savelist)
    
    # 保存到本地excel
    df.to_excel(filename, index=False)


if __name__ == '__main__':
    
    company_name_list = ['腾讯', '阿里巴巴', '字节跳动', '腾讯']
    number_list = [i for i in range(4)]
    x = [company_name_list, number_list]
    savexcl(x, 'company_name_li.xlsx')
'''
import pandas as pd


# python+pandas 保存list到本地
def deal():
    # 二维list
    company_name_list = [['腾讯', '北京'], ['阿里巴巴', '杭州'], ['字节跳动', '北京']]

	# list转dataframe
    df = pd.DataFrame(company_name_list, columns=['company_name', 'local'])

	# 保存到本地excel
    df.to_excel("company_name_li.xlsx", index=False)


if __name__ == '__main__':
    deal()

'''