try:
    with open('test.txt') as file:
        print(file.read())
except FileNotFoundError as fne:
    print(fne)
    
# with는 close()까지 해준다