import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector() # 얼굴 탐지 모델 (detection)
sp = dlib.shape_predictor('teamproject/simple_face_recognition-master/models/shape_predictor_68_face_landmarks.dat') # 얼굴 랜드마크 (눈,코,입 등)탐지 모델(face landmark detection)
facerec = dlib.face_recognition_model_v1('teamproject/simple_face_recognition-master/models/dlib_face_recognition_resnet_model_v1.dat') # 얼굴인식모델


# 얼굴을 찾는 함수
def find_faces(img): # rgb로 인풋 받고 
    dets = detector(img, 1) # 찾은 얼굴 결과물을 dets 변수에 저장

    if len(dets) == 0: # 얼굴을 못찾으면 빈 배열을 반환
        return np.empty(0), np.empty(0), np.empty(0)
    
    rects, shapes = [], [] # shape은 68개의 점, 랜드마크를 구하는 함수
    
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    for k, d in enumerate(dets): # 얼굴 갯수만큼 루프를 구함
        rect = ((d.left(), d.top()), (d.right(), d.bottom())) # rect 변수에 얼굴 왼쪽, 위, 오른쪽, 아래 좌표를 넣어줌
        rects.append(rect)
        
        shape = sp(img, d) # 이미지와 사각형  rect 을 얼굴 랜드마크 탐지 모델에 넣어주면 shape 변수에 68개 점이 나옴

        #convert dlib shape to numpy array

        for i in range(0,68): # 랜드마크 결과물들도 shapes 라는 리스트에 넣어줌
            shapes_np[k][i]=(shape.part(i).x, shape.part(i).y)
        shapes.append(shape)
    return rects, shapes, shapes_np

# 얼굴을 인코딩 하는 함수 - 눈,코,입,귀 등의 랜드마크를 인코딩하여 128개의 벡터로 변환
def encode_faces(img, shapes): 
    face_descriptors = []
    for shape in shapes: # 랜드마크의 집합 크기 만큼 루프 돌면서 
        face_descriptors = facerec.compute_face_descriptor(img, shape) # facerec 모델을 돌려줌/compute_face_descriptor 메소드를 써줌/ 전체 이미지와 랜드마크가 인풋.
        face_descriptors.append(np.array(face_descriptor)) 
    
    return np.array(face_descriptors) # 벡터값 반환

img_paths = { # 사용자들 이미지 경로
    'neo': 'teamproject/simple_face_recognition-master/img/neo.jpg',
    'trinity': 'teamproject/simple_face_recognition-master/img/trinity.jpg',
    'morpheus': 'teamproject/simple_face_recognition-master/img/morpheus.jpg',
    'smith': 'teamproject/simple_face_recognition-master/img/smith.jpg'
}

descs = { # 계산할 결과를 저장할 변수
    'neo': None,
    'trinity': None,
    'morpheus': None,
    'smith': None
}

for name, img_path in img_paths.items(): # 이미지 경로만큼 루프를 돌면서 
    img_bgr = cv2.imread(img_path) # 이미지 불러서
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # bgr을 rgb로 바꿈

    _, img_shapes, _ = find_faces(img_rgb) # 얼굴 찾는 함수로 랜드마크를 받아오고
    descs[name] = encode_faces(img_rgb, img_shapes)[0] # 인코드 함수에 전체 이미지와 각 사람의 랜드마크를 넣어줍니다/ 각 사람의 이름에 맞게 저장

np.save('teamproject/simple_face_recognition-master/img/descs.npy', descs) # 결과값을 넘파이 세이브로 저장
print(descs)

# compute input
img_bgr = cv2.imread('teamproject/simple_face_recognition-master/img/matrix5.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes) # 얼굴 인코드한 결과를 descriptors로 받아옴


# Visualize output
fig, ax = plt.subplots(1, figsize=(20, 20))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors): # descriptors만큼 루프를 돈다
    
    found = False
    for name, saved_desc in descs.items(): # 아까 저장한 각 사람들의 인코딩한 것 (descs)에 
        dist = np.linalg.norm([desc] - saved_desc, axis=1)

        if dist < 0.6:
            found = True

            text = ax.text(rects[i][0][0], rects[i][0][1], name,
                    color='b', fontsize=40, fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects.Normal()])
            rect = patches.Rectangle(rects[i][0],
                                 rects[i][1][1] - rects[i][0][1],
                                 rects[i][1][0] - rects[i][0][0],
                                 linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)


            break
    
    if not found:
        ax.text(rects[i][0][0], rects[i][0][1], 'unknown',
                color='r', fontsize=20, fontweight='bold')
        rect = patches.Rectangle(rects[i][0],
                             rects[i][1][1] - rects[i][0][1],
                             rects[i][1][0] - rects[i][0][0],
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.axis('off')
plt.savefig('teamproject/simple_face_recognition-master/result/output.png')
plt.show()