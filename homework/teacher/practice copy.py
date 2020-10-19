
# numpy로 바꾸기
# 200장을 400장으로 증폭, 별도의 폴더에 집어 넣기 ( 400장이 들어간 폴더 하나)
# numpy에 200장 데이터를 (200,150,150,3)를 저장
# y는 (200,)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

data = 'F:\Study\homework/teacher\cslee2'
path = 'teacher\cslee3'

gen = ImageDataGenerator(rescale= 1./255,
                         horizontal_flip=True,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         fill_mode='nearest'
       )

data_gen = gen.flow_from_directory('F:\Study\homework/teacher\cslee2',
                                   target_size=(150,150),
                                   batch_size=30,
                                #    save_to_dir = path,
                                #    save_prefix = 'babo',
                                #    save_format='jpg',
                                   class_mode='binary')

print('test_x.shape: ', data_gen[0][0].shape) # (30, 150, 150, 3)
print('test_y.shape: ', data_gen[0][1].shape) # (30,)
test_x = data_gen[0][0]
test_y = data_gen[0][1]

import numpy as np
# np.save('./teacher/cslee2', arr=test_x)
# np.save('./teacher/cslee2', arr=test_y)


'''
# 증식할 이미지 개수 지정
augment_num = 400
path = 'teacher\cslee2'


for i in range(200):
    image_filename = str(i) + '.jpg'
    gen.fit(test_x)
    for x, y in zip (
        gen.flow( test_x, #input
                       test_y,  # y
                       batch_size = 50,
                       shuffle = True,
                       save_to_dir = path,
                       save_prefix = '',
                       save_format='jpg'),
                       range(augment_num)):
        pass

print(' end')    

 # flow method: Takes data & label arrays, generates batches of augmented data.
'''
 

