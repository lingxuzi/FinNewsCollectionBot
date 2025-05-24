import qstock

def get_qstock_news():
    lastday, datenow = None, None
    for _ in range(3):
        try:
            cailian_news = qstock.news_data(news_type=None,start=lastday,end=datenow,code=None)
            cailian_news = '\n\n'.join(cailian_news['内容'].values)
            break
        except Exception as e:
            cailian_news = ''

    for _ in range(3):
        try:
            cctv_news = qstock.news_data(news_type='cctv',start=lastday,end=datenow,code=None)
            cctv_news = '\n\n'.join(cctv_news['content'].values)
            break
        except Exception as e:
            cctv_news = ''

    for _ in range(3):
        try:
            js_news = qstock.news_data(news_type='js',start=lastday,end=datenow,code=None)
            js_news = '\n\n'.join(js_news['content'].values)
            break
        except Exception as e:
            js_news = ''


    return '\n\n'.join([cailian_news,cctv_news,js_news])

if __name__ == '__main__':
    get_qstock_news()