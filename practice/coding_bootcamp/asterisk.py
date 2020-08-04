# ⊙ asterisk의 언패킹 기능

# dictionary로 바꿔줌
def foo(**kwargs):
    print(kwargs)
foo(bar='bar')
# {'bar': 'bar'}


# Tuple로 바꿔줌
def foo(*args):
    print(args[0])
    print(args[1])
foo('bar',99)
# bar
# 99