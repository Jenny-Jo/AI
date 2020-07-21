
# keyword 인수
def book_info(title, price, publisher):
    print('도서명: ', title)
    print('가격 : ', price)
    print('출판사 : ', publisher)
book_info(title='coding bootcamp', publisher='길벗', price='20000')

# 도서명:  coding bootcamp
# 가격 :  20000
# 출판사 :  길벗


# 정의되지 않은 키워드 인수
def foo(**kwargs):
    print(kwargs)
foo(bar='bar', hoge='hoge', num=999)
# {'bar': 'bar', 'hoge': 'hoge', 'num': 999}