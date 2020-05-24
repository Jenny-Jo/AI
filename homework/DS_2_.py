for i in [1,2,3,4,5]:
    print(i)
    for j in [1,2,3,4,5] :
        print(j)
        print(i+j)
        print("=====")
    print(i)
    print("===+++=")
print("done looping")

a = 2+\
    3

for i in [1,2,3]:
    
    print(i)


#2.6  함수

def double(x) :
    return x*2

def apply_to_one(f):
    return f(1)

my_double = double
x = apply_to_one(my_double)
y = apply_to_one(lambda x: x+4)

another_double = lambda x: 2*x
def another_double(x) :
    return 2*x

def my_print(message = "my default message") :
    print(message)
my_print("Hello")
print()

def full_name(first = "what's-shis-name", last="something") :
    return first + " " + last


print(full_name("Jo","Jenny"))
print(full_name("Jenny"))
full_name(last="Jo")

# 2.7 문자열
tab_string = "/t"
print(len(tab_string))

first_name = "Joel"
last_name = "Grus"
full_name = first_name + "" + last_name
full_name2="{0}{1}".format(first_name, last_name)
full_name3 = f"{first_name},{last_name}"

#2.8 예외처리
try:
    print(0/0)
except ZeroDivisionError :
    print("cannot divide by zero")

#2.9 리스트
x= [-1,1,2,3,4,5,6,7,8,9]
# every_third = [::3] #[-1,3,6,9]
five_to_three = x[5:2:1]
print(five_to_three) #[]?? #[5,4,3]

x = [1,2,3]
x.extend([4,5,6]) #[1, 2, 3, 4, 5, 6]
print(x)

#2.10 튜플

my_list = [1,2]
my_tuple = (1,2)
other_tuple = 3,4
my_list[1]  = 3 # [1,3] 1번째 2에 3이 들어가므로

try:
    my_tuple[1] =3
except TypeError:
    print("cannot modify a tuple")

def  sum_and_product(x,y):
    return (x+y),(x*y)

sp= sum_and_product(2,3) #(5,6)
s, p = sum_and_product(5,10) # (s=15,p=50)

#2.11 dictionary
empty_dict ={}
grades = {"Joel":80, "Tim":95}
joels_grade = grades["Joel"]

try:
    kates_grade = grades["Kate"]
except KeyError:
    print("no grade for Kate")

jeols_has_grade = "Joel" in grades
kate_has_grade = "Kate" in grades #False

kattes_grade = grades.get("Kate",0) #0

grades["Tim"] = 90 # 대체
grades["Kate"] = 100 # 추가
num_students = len(grades) #3

instagram = {
    "user" : "Jenny",
    "text" : "Data Science is awesome!",
    "like_count" : 100,
    "hashtags":["#data","#science","#yolo"]
}

instagram_keys = instagram.keys()
print(instagram_keys) # dict_keys(['user', 'text', 'like_count', 'hashtags'])
instagram_values = instagram.values()
instagram_items = instagram.items()
print(instagram_items) #dict_items([('user', 'Jenny'), ('text', 'Data Science is awesome!'), ('like_count', 100), ('hashtags', ['#data', '#science', '#yolo'])])

# "user" in instagram_keys()
"user" in instagram # Tuple is better than finding in list
"Jenny" in instagram_values # True



#2.11.1 defaultdict
# word_counts = {}
# for word in document :
#     if word in word_counts:
#         word_counts[word] += 1
#     else:
#         word_counts[word]=1

# word_counts = {}
# for word in document :
#     try:
#         word_counts[word] += 1
#     except KeyError :
#             word_counts[word]=1

# word_counts = {}
# for word in document:
#     previous_count = word_counts.get(word,0)
#     word_counts[word] = previous_count +1

# from collections import defaultdict

# word_counts = defaultdict(int)
# for word in document :
#     word_counts[word] += 1

# dd_list = defaultdict(list)
# dd_list[2].append(1)

# dd_dict = defaultdict(dict)
# dd_dict = ["Joel"]["city"] = "seatle"

# dd_pair = defaultdict(lambda : [0,0])
# dd_pair = [2][1]= 1

#2.12 counter
from collections import Counter
c = Counter([0,1,2,0])
print(c) #Counter({0: 2, 1: 1, 2: 1})

#2.13 set
s = set()
s.add(1)
s.add(2)
s.add(2)
x = len(s)
y = 2 in s 
z = 3 in s


# stopwords_list= ["a","an","at"] + hundreds_of_other_words + ["yet", "you"]

# "zip" in stopwords_list

# stopwords_set = set(stopwords_list)
# "zip" in stopwords_set

item_list = [1,2,3,1,2,3]
num_items = len(item_list)
item_set = set(item_list)
num_distinct_items = len(item_set)
distinct_item_list = list(item_set)

#2.14  흐름제어
if 1>2:
    message = "if only 1 were greater than two"
elif 1>3:
    message = "elif stands for 'else if'"
else :
    message = "when all else fails use else"

for x in range(10):
    print(f"{x} is less than 10")

for x in range(10):
    if x == 3:
        continue
    if x == 5:
        break
    print(x)

s = some_function_that_returns_a_string()
if s:
    first_char = s[0]
else:
    first_char = s and [0]

safe_x = x or 0

safe_x = x if x is not None else 0

all([True,1,{3}])
all([True,1, {}])
any([True,1, {}])
all([])
any([])

#2.16  정렬
x = [4,1,2,3]
y = sorted(x)
x.sort()

x = sorted([-4,1,-2,3], key = abs, revers= True)

wc = sorted(word_counts.items(), key = lambda word_and_count:word_and_count[1],reverse = True)

#2.17 list comprehension
even_numbers = [x for x in range(5) if x % 2 == 0]
squares = [x*x for x in range(5)]
even_squares = [x*x for x in even_numbers]

square_dict = {x: x*x for x in range(5)}
square_set = { x*x for x in [1,-1]}

zeros = [0 for _ in even_numbers]

pairs = [(x,y)
        for x in range(10)
        for y in range(10)]

increasing_paris = [(x,y)
                    for x in range(10)
                    for yu in range(x+1, 10)]

#2.18 자동테스트, assert
assert 1 + 1 ==2
assert 1 + 1 ==2. "1 + 1 should equal 2 but didn't"

def smallest_item(xs):
    return min(xs)

assert smallest_item([10,20,5,40]) ==5
assert smallest_item([1,0,--1,2]) == -1

def smallest_item(xs):
    assert xs, "empty list has no smallest item"
    return min(xs)

#2.19  객체 지향 프로그래밍
 class CountingClicker :
     
def__int__ (self, count = 0):
    self.count = count(x)

clicker1 = CountingClicker()
clicker2 = CountingClicker(100)
clicker3 = CountingClicker(count = 100)

def __repr__(self):
    return f"CountingClicker(count={self.count})"

def click(self, num_times = 1):
    self.count += num_times
def read(self):
    return self.count
def reset(self):
    self.count = 0

clicker = CounterClicker()
assert clicker.read() == 0, "clicker should start with 0"
clicker.click()
clicker.click()
assert clicker.read() == 2, "after two clicks, clicker should have count 2"
clicker.reset()
assert clicker.read() == 0, "after reset, clicker should be back to 0"

class NoResetClicker(CounterClicker):
    def reset(self):
        pass

clicker2 = NoResetClicker()
assert clicker2.read() == 0
clicker2.click()
assert clicker2.read() == 1
clicker2.reset()
assert clicker2.read() == 1, "reset shoudln't do anything"

#2.20 이터레이터, 제너레이터

def generate_range(n):
    i = 0
    while i < 2:
        yield i
        i += 1

for i in generate_range(10):
    print(f"i:{1}")

def natural_numbers():
    n = 1
    while True:
        yield n
        n += 1

evens_below_20 = (i for i in generate_range(20) if i % 2 == 0)

data = natural_numbers()
evens = (x for x in data if x % 2 == 0 )
even_squares = (x **2 for x in evens)
even_squares_ending_in_six = (x for x in even_squares if x %10 == 6)

names = ["Alice", "Bob", "Charlie","Debbie"]
for i, name in enumerate(names):
    print(f"name{1} is {name}")

# 2.21 난수생성
import random.seed(10)

four_uniform_randoms = [random.random() for _ in range(4)]

random.seed(10)
print(random.random())
random.seed(10)
print(random.random())

random.randrange(10)
random.randrange(3,6)

up_to_ten = [1,2,3,4,5,6,7,8,9,10]
random.shuffle(up_to_len)
print(up_to_ten)

my_BF = random.choice(["Alice","Jen","Lucy"])

lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)

four_with_replacement = [random.choice(range(10)) for _ in range(4)]
print(four_with_replacement)

#2.2 정규표현식
 
import re

re_ex = [
    not re.match("a","cat"),
    re.search("c", "dog"),
    not re.search("c","dog"),
    3 == len(re.split("[ab]", "carbs")),
    "R-D-" == re.sub("[0-9]","-", "R2D2")
]

assert all(re_examples), "all the regex examples should be True"

# 2.24 zip 과 인자 언패킹

list1 = ['a','b','c']
list2 = [1,2,3]

[pair for pair in zip (list1, list2)]

pairs = [('a',1), ('b',2), ('c',3) ]
letters, numbers = zip(*pairs)
letters, numbers = zip(('a',1), ('b',2), ('c',3))

def add(a,b) : return  a + b
add(1,2) 
try:
    add([1,2])
except TypeError:
    print("add expects two inputs")

#2.25 args , kwargs
def doubler(f):
    def g(x):
        return 2 * f(x)
        return g

def f1(x):
    return x + 1

g = doubler(f1)
assert g(3) == 8, "(3 +1)* 2 should equal 8"
assert g(-1) == 0 , "(-1+1)*2 should equal 0"

def f2(x,y):
    return x+y
g = doubler(f2)
try:
    g(1,2)
except TypeError:
    print("as defined, g only takes one argument")

def magic(*args, **kwargs):
    print("unnamed args:", args)
    print("keyword args:",kwargs)
magic(1,2, key="word", key2="word2")


def other_way_magic(x, y, z):
    return x + y + z

x_y_list = [1,2]
z_dict = {"z":3}
assert other_way_magic(*x_y_list, **z_dict) == 6, "1 + 2+ 3 should be 6"

def doubler_correct(f):
    def g(*args, **kwargs):
        return 2 *f(*args, **nkwargs)
    return g

g = doubler_correct(f2)
assert g(1,2) == 6, "doubler should work now "

# 2.26 type annotation

def add(a, b):
    return a + b 

assert add(10,5) == 15, "+ is valid for numbers"
assert add([1,2],[3]) == [1,2,3] "+ is valid for lists"
assert add("Hi","there") == "Hi there", "+ is valid for strings"

try:
    add(10,"five")
except TypeError:
    print("cannot add an int to a string")

#2.26.1
from typing import List

def total(xz: List[loat]) ->float:
    return(total)

from typing import Optional

values: List[int] = []
best_so_far: Optional[float] = None

from typing import Dict, Iterable, Tuple
counts: Dict[str, int] = {'data':1, 'science' :2}

if lazy : 
    evens: Iterable[int] = (x for x in range(10) if x%2 ==0)
else:
    evens = [0,2,4,6,8]
triple: Tuple[int,float, int] = (10, 2.3, 5)

from typing import Callable
def twice (repeater:Callable[[str,int],str], s: str) -> str :
    return repeater(s,2)

def comma_repeater(s: str, n:int) -> str:
    n_copies = [s for _ in range(n)]
    return ', '.join(n_copies)

assert twice(comma_repeater, "type hints") == "type hints, type hints"

Number = int
Numbers = List[Number]

def total(xs: Numbers) -> Number:
    return sum(xs)
     


