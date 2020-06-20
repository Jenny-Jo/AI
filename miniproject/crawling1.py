from selenium import webdriver
import time

def get_comments(url) : 

    # 웹드라이버
    driver = webdriver.Chrome('./chromedriver.exe')
    driver.implicitly_wait(3)
    driver.get(url)

    # 더보기 계속 클릭
    while True : 
        try:
            더보기 = driver.find_element_by_css_selector('a.u_cbox_btn_more')
            더보기.click()
            time.sleep(1)
        except :
            break

    print('끝')

    # 댓글 추출
    contents = driver.find_elements_by_css_selector('span.u_cbox_contents')
    for content in contents:
        print(content.text)

    # # 작성자 추출
    # while True :
    #     try :
    #         click_ID = driver.find_element_by_css_selector('u_cbox_btn_totalcomment')
    #         click_ID.click()
    #         time.sleep(1)
    #         nicks = driver.find_element_by_css_selector('u_cbox_userinfo_meta_nickname')
    #         for nick in nicks :
    #             print(nick.text)
    #     except :
    #         break

    driver.quit()

    return contents

if __name__ =='__main__' :
    url ='https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=104&oid=052&aid=0001452183&m_view=1&includeAllCount=true&m_url=%2Fcomment%2Fall.nhn%3FserviceId%3Dnews%26gno%3Dnews052%2C0001452183%26sort%3Dlikability'
    comments_data = get_comments(url)

    import pandas as pd
    col = ['댓글']
    data_frame = pd.DataFrame(comments_data, columns = col)
    data_frame.to_excel('news.xlsx', sheet_name='뉴스기사제목', startrow=0, header=True)