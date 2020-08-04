# inheritance 상속
class A:
    name = 'Class A'
class B(A):
    pass
b = B()
print(b.name)

# override -부모 클래스에 있는 기능을 덮어쓰는 것

class A :
    def hello(self):
        print('Class A says Hello')
class B(A):
    def hello(self):
        print('Class B says Hello')

b = B()
b.hello()