# << 1.내장함수와 메서드 >>
# 1. 내장함수-------------------------------
# len()는 ()내 '객체'의 길이나 요소 수 반환
vege = "potato"
n = [4,5,2,7,6]
print(len(vege))
print(len(n))
# len(3)
# len(1.5)
# len(True)

# 2. method--- .sort() --------값에 대해 처리-----------
number = [ 4,6,2,7,1]

number.sort()
print(number)           #[1, 2, 4, 6, 7]
#원래 리스트 내용까지 바꿔버리는 sort() method 는 파괴적 메서드라고 함

# print(sorted(number)) # [1, 2, 4, 6, 7]
# print(number)         # [4, 6, 2, 7, 1]

# 3. 변수.upper() , 변수.count()----------------------------
# .upper()은 () 안에 들어 있는 '문자열'에 요소가 몇 개 포함되어 있는지 알려주는 메서드
a = 'abcdbbb'
print(a.upper())
print(a.count('b'))

animal = 'elephant'
animal_big= animal.upper()
print(animal_big)
print(animal.count('e'))


# 4. 변수.format() --------------------------------------------
print("{} is {}".format('banana','yellow'))
fruit = 'apple'
color = 'red'
print("{} is {}".format(fruit, color))

# 5.리스트형 메서드 index/ 변수.index()------------------------
a = ['a','b','c','d','e']
print(a.index('a'))
print(a.count('d'))

n = [3,4,5,2,6,7,2,4,3,4]
print(n.index(3))
print(n.count(3))

# 6.  variable.sort()--------------------------
# list형에 사용/ 오름차순 정렬/ 리스트 내용 변경

n = [53,25,46,13,5,7,89]
n.sort()
print(n)
n.reverse()
print(n)

# << 함수 >>
def  함수명(): # ()안에는 인수 들어감
    print('처리하기')
함수명()

def introduce():
    print('I am Jenny')
introduce()

# 1. 인수
def cube_cal(n):
    print(n**3)
cube_cal(4)
# 인수가 인수로 지정된 변수에 대입되기 때문에

# 2. 복수 개의 인수
def introduce(n, age):
    print(n+'입니다.'+str(age)+'살입니다')
introduce('홍길동',18)

def fruit(season, fruit):
    print(season+'에는',fruit+'을/를 먹습니다')
fruit('여름', '수박')

# 3. 인수의 초깃값??? 모르겠음

# 4. return
def bmi(w,h):
    return w/h**2
print(bmi(52,1.6))

# 5. import// package.module
import time
now_time = time.time()
print(now_time)

from time import time # from package import module
now_time = time()
print(now_time)

# PyPI = Python Package Index
from time import time
now_time = time()
print(now_time)

# << class >>
'''
class MyProduct:
    def __init__(self,name,price,stock):
        self.name =name
        self.price = price
        self.stock = stock
        self.sales = 0

# product1 = MyProduct("cake",500,20)
# print(product1.stock)

    def buy_up(self,n):
        self.stock += n

    def sell(self,n):
        self.stock -=n
        self.saled += n*self.price
    def summary(self):
        message = "called summary().\n name:" + self.name + \
            "\n price: " +str(self.price) + \
            "\n stock: " +str(self.stock) + \
            "\n sales: " +str(self.sales)
    def get_name(self):
        return self.name
    def discount(self, n):
        self.price-=n

product_2 = MyProduct('phone', 30000, 100)
product_2.discount(5000)
product_2.summary()
'''

class MyProduct:
    def __init__(self,name,price,stock):
        self.name =name
        self.price = price
        self.stock = stock
        self.sales = 0

    def summary(self):
        message = "called summary().\n name:" + self.name + \
            "\n price: " +str(self.price) + \
            "\n stock: " +str(self.stock) + \
            "\n sales: " +str(self.sales)
        print(message)

    def get_name(self):
        return self.name

    def discount(self, n):
        self.price-=n

class MyProductSalesTax(MyProduct):
    def __init__(self, name, price, stock, tax_rate):
        super().__init__(name, price, stock)
        self.tax_rate = tax_rate
    def get_name(self):
        return self.name + "(소비세 포함)"
    def get_price_with_tax(self):
        return int(self.price*(1+self.tax_rate))
    def summary(self):
        message = 'called summary().\n name:' + self.get_name()+\
                  '\n price:' + str(self.get_price_with_tax()+0) + \
                  '\n stock:' + str(self.stock) + \
                  '\n sales: '+ str(self.sales)
        print(message)

product_3 = MyProductSalesTax("phone", 30000, 100, 0.1)
print(product_3.get_name())
print(product_3.get_price_with_tax())
product_3.summary()

# 문자열 포맷 지정
pai = 3.141592
print("원주율은 %f" % pai)
print("원주율은 %.2f" % pai)

def bmi(h,w):
    return w/h**2
h = 1.6
w = 52
my_bmi = bmi(h,w)
print("bmi는 %.4f입니다."%bmi(1.6, 52))

# 연습문제
def check_character(object, character):
    return object.count(character)
print(check_character([1,3,5,6,7,3,23,5,7,8,3], 3))
print(check_character('a;sldkjfieujfija;slkdfj','d'))

# 종합문제
def binary_search(numbers, target_number):
    low = 0
    high = len(numbers)
    while low <=high:
        middle = (low+high)//2
        if numbers[middle] == target_number:
            print("{1}은 {0}째에 있습니다.".format(middle, target_number))
            break
        elif numbers[middle]< target_number:
            low = middle + 1
        else:
            high = middle - 1
numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13]
target_number = 11
binary_search(numbers, target_number)
    
        