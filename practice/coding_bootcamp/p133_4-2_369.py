def game369(v):
    if v % 3 == 0:
        return ('짝')
    elif '3' in str(v):
        return ('짝')
    # else:
    return(v)

v = int(input('input하시오'))
a = game369(v)
print(a)

    