import requests
from lxml import html
url='http://www.sse.com.cn/disclosure/credibility/supervision/inquiries/' #需要爬数据的网址
page=requests.Session().get(url)
tree=html.fromstring(page.text)
result=tree.xpath('/html/body/div[8]/div[2]/div[2]/div[2]/div/div/div/div[1]/div[2]/div/div[2]/div/table/tbody') #获取需要的数据
print(result)#/html/body/div[8]/div[2]/div[2]/div[2]/div/div/div/div[1]/div[2]/div/div[2]/div/table/tbody