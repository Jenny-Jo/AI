# numpy로 바꾸기
# 200장을 400장으로 증폭, 별도의 폴더에 집어 넣기 ( 400장이 들어간 폴더 하나)
# numpy에 200장 데이터를 (200,150,150,3)를 저장
# y는 (200,)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# a = ImageDataGenerator()
# a.flow_from_directory() # batch_size 의미 - No. of images to be yielded from the generator per batch.
#                                             # generator에서 한번에 가져오는 이미지 갯수

# a.next() vs a[0]


print('-----------------')

b = ImageDataGenerator(rescale=1./255,          # 0~1변환 / 정규화한다.
                                   horizontal_flip=True,    # vertical_flip : 주어진 이미지를 수평 또는 수직으로 뒤집는다.
                                   width_shift_range=0.1,   # 정해진 범위 안에서 그림을 수평 또는 수직으로 랜덤하게 평행 이동
                                   height_shift_range=0.1,  #
                                   # rotation_range         #  정해진 각도만큼 이미지를 회전
                                   fill_mode='nearest'      # 이미지를 축소 또는 회전하거나 이동할때 새익는 빈 공간을 어떻게 할지.
                                                            # nearest  가장 비슷한 색으로 채운다.
                                   )

c = b.flow_from_directory('teacher/down/test',
    target_size=(150, 150),
    batch_size=200,           # 원래 5
    class_mode='binary') # categorical, input

print('c.next() :',c.next())
print('-'*50)
print('c[0]: ',c[0])

print('-----------------------')
# https://python.bakyeono.net/chapter-7-4.html
'''
c.next()

반복 가능한 데이터: iter() 함수로 반복자를 구할 수 있는 데이터
반복자: next() 함수로 값을 하나씩 꺼낼 수 있는 데이터
iter() 함수: 반복 가능한 데이터를 입력받아 반복자를 반환하는 함수
next() 함수: 반복자를 입력받아 다음 출력값을 반환하는 함수
''''''
>>> it = iter([1, 2, 3])  # [1, 2, 3]의 반복자 구하기
>>> next(it)              # 반복자의 다음 요소 구하기
1

>>> next(it)              # 반복자의 다음 요소 구하기
2

>>> next(it)              # 반복자의 다음 요소 구하기
3

>>> next(it)              # 더 구할 요소가 없으면 오류가 발생한다
StopIteration'''

'''
생성기는 반복자의 한 종류다.
생성기는 yield 문이 포함된 함수를 실행하여 만들 수 있다.
yield 문이 포함된 함수를 실행하면 생성기가 반환된다. 생성기를 next() 함수에 전달해 실행시키면 함수의 본문이 실행된다.
yield 문은 값을 내어준 후 생성기의 실행을 일시정지한다. next() 함수가 실행되면 정지했던 위치에서부터 다시 실행이 이어진다.'''
'''
>>> def abc():  # ❶ 생성기를 반환하는 함수 정의하기
...     "a, b, c를 출력하는 생성기를 반환한다."
...     yield 'a'
...     yield 'b'
...     yield 'c'
... 
>>> abc()       # ❷ 생성기 만들기


>>> abc_generator = abc()  # ❶ 생성기 만들기
>>> next(abc_generator)    # ❷ 생성기의 다음 값 꺼내기
'a'

>>> next(abc_generator)
'b'

>>> next(abc_generator)
'c'

>>> next(abc_generator)    # ❸ 더 구할 요소가 없으면 오류가 발생한다
StopIteration


''''''
>>> def one_to_infinite():
...     """1 - 무한대의 자연수를 순서대로 내는 생성기를 반환한다."""
...     n = 1                            # n은 1에서 시작한다
...     while True:                      # ❶ 무한 반복
...         yield n                      # ❷ 실행을 일시정지하고 n을 반환한다
...         n += 1                       # n에 1을 더한다
... 
>>> natural_numbers = one_to_infinite()  # ❸ 생성기를 만들어
>>> next(natural_numbers)                #    수를 무한히 꺼낼 수 있다
1

>>> next(natural_numbers)
2
'''

# c[0]

# Returns

# A DirectoryIterator yielding tuples of (x, y) 
# where x is a numpy array containing a batch of images with shape (batch_size, *target_size, channels) and 
# y is a numpy array of corresponding labels.

