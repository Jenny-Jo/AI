#1
dict_var = {'price':1000, 'name':'양말', 'stock':50, 'code':'CY001'}
dict_var1= dict(price=1000, name='양말', stock=50, code='CY001')

for key,value in dict_var.items():
    print(key+' > '+ str(value))

#2

def get_sum (start, end):
    a = 0
    for i in range(start,end):
        a+=i
    # return a
    print(a)

get_sum(start=1, end=10)

# 정답

def get_sum(**kargs):
    start = kargs['start']
    end = kargs['end']
    result = 0
    for v in range(start, end+1):
        result +=v
    return result

val = get_sum (start = 1, end = 10)
print(val)