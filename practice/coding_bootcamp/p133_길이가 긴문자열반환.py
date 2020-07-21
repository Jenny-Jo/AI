# def compare_string(a,b):
#     a = str(input('a'))
#     b = str(input('b'))
#     if len(a)<len(b):
#         return b
#     return a

# a=input('a')
# b=input('b')
# compare_string(a,b)

def compare_string(a,b):
    if len(a)<len(b):
        return b
    return a

result = compare_string('aaa','bbb')
print(result)