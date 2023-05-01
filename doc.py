import requests
from bs4 import BeautifulSoup
import pandas as pd
import traceback
from urllib.request import urlopen, Request



def item_search(item, limit, page):
    stocknews = f"https://www.thestar.com.my/search/?q={item}&qsort=oldest&qrec={limit}&qstockcode=&pgno={page}"

    html = requests.get(stocknews).text

    soup = BeautifulSoup(html, 'html.parser')
    return soup

def get_details(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')

    content = soup.find('div', {'id':'story-body'})
    if content:
        content.get_text(strip=True)
        return content
    else:
        return ""
def star_new_crawler(page, search_query, limit):

    title = []
    links = []
    premium = []
    new_type = []
    contents = []
    publishedDate = []
    while True:
        print(page)
        try:
            result = item_search(search_query, limit, page)
            title += [x.get_text(strip=True) for x in result.find_all("h2", {"class": "f18"})]
            links += [x.find('a', {"data-content-type": "Article"})['href'] for x in
                     result.find_all("h2", {"class": "f18"})]
            premium += [x.get_text(strip=True) for x in result.find_all("span", {"class": "biz-icon"})]
            new_type += [x.get_text(strip=True) for x in result.find_all("a", {"class": "kicker"})]
            contents += [get_details(x) for x in links]
            publishedDate += [x.get_text(strip=True) for x in result.find_all("span", {"class": "timestamp"})]
         
            if len(title) == 0:
                break
        except Exception as e:
            print(e)
            traceback.print_exc()
    
        page += 1

    pd.DataFrame({'new_type': new_type, 'title': title, 'premium': premium, 'links': links, 'published_data': publishedDate,
                 'contents': contents}).to_excel(f'{search_query}.xlsx', index=False)

if __name__ == '__main__':
    page = 1
    search_query = 'AAPL'
    limit = 30
    star_new_crawler(page, search_query, limit)
