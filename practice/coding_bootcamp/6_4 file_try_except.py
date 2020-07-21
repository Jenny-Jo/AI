try:
    file = open('python.txt', 'r')
except FileNotFoundError as fne:
    print('파일을 찾을 수 없습니다. 확인해주세요')

total=int(input('몇명?: '))
amount = int(input('얼마나? : '))

def dutch(total,amount):
    try:
        return amount/total
    except:
        pass

x = dutch(total,amount)
print(x)