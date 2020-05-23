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

"user" in instagram_keys()
"user" in instagram # Tuple is better than finding in list
"Jenny" in instagram_values # True

