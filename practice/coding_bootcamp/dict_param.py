# 키워드 인수??
# 정의되지 않은 키워드 인수
def foo(**kwargs):
    print(kwargs)
foo(bar='bar', hoge='hoge',num=999)
# {'bar': 'bar', 'hoge': 'hoge', 'num': 999}  
