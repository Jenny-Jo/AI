import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector() # 얼굴탐지모델
sp = dlib.shape_predictor('teamproject/simple_face_recognition-master/models/shape_predictor_68_face_landmarks.dat') # 랜드마크 탐지 모델
facerec = dlib.face_recognition_model_v1('teamproject/simple_face_recognition-master/models/dlib_face_recognition_resnet_model_v1.dat')#얼굴인식모델

def find_faces(img): # 얼굴 찾는 함수
    dets= detector(img, 1)

    if len(dets) == 0: # 얼굴 못찾으면 빈 배열 반환
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int ) # 68개의 랜드마크 구하는 함수
    for k, d in enumerate(dets): # 얼굴마다 루프를 돌아
        rect = ((d.left(), d.top()), (d.right(), d.bottom())) # 얼굴 왼 위 오른 아래 좌표를 넣어준다
        rects.append(rect)
        
        # landmark
        shape = sp(img, d) # 이미지, 사각형 넣으면 68개 랜드마크 나와

    # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape) # 랜드마크 결과물을 쌓아줌
    
    return rects, shapes, shapes_np

def encode_faces(img, shapes): # 얼굴인코드 -68개 랜드마크를 인코더에 넣으면 128개의 벡터가 나오고, 그걸로 사람 얼굴 구분. 같은 사람인지 아닌지
    face_descriptors=[]
    for shape in shapes: # 랜드마크 집합만큼 루프 돌리면서 
        face_descriptor = facerec.compute_face_descriptor(img, shape) # 얼굴 인코딩 (전체이미지, 각 사람들의 랜드마크)
        face_descriptors.append(np.array(face_descriptor)) # 결과값을 넘파이로 바꾸면서 face_descriptors에 쌓아준다

    return np.array(face_descriptors)

# Compute Saved Face Descriptions - 미리 저장한 얼굴들의 인코딩한 데이터 저장

img_paths = {
    'neo': 'teamproject/simple_face_recognition-master/img/neo.jpg',
    'trinity':'teamproject/simple_face_recognition-master/img/trinity.jpg',
    'morpheus':'teamproject/simple_face_recognition-master/img/morpheus.jpg',
    'smith':'teamproject/simple_face_recognition-master/img/smith.jpg'
}
# 이미지는 저장소에 저장, 인코딩된 값을 저장했다가 인풋이 들어왔을 때 인코드된 값을 꺼내다가 쓸거임
descs = { # 계산한 결과를 저장할 변수
    'neo':None,
    'trinity':None,
    'morpheus':None,
    'smith':None
}

for name, img_path in img_paths.items(): # 이미지 path만큼 루프 돌면서 
    img_bgr = cv2.imread(img_path) # 이미지 로드 
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # bgr을 rgb로 바꿈

    _, img_shapes, _ = find_faces(img_rgb) # rgb로 바꾼 이미지에서 얼굴 찾아서 랜드마크를 받아옴
    descs[name]=encode_faces(img_rgb, img_shapes)[0] # encode_faces 함수에 전체 이미지와 각 사람의 랜드마크 넣어줌/각 사람의 이름에 맞게 저장해줌

np.save('teamproject/simple_face_recognition-master/img/descs.npy', descs) # 넘파이 파일로 저장

print(descs)

# compute input
img_bgr = cv2.imread('teamproject/simple_face_recognition-master/img/matrix2.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes) # 인코드한 결과를 descriptors에 받아옴

# visualize output / 결과값을 뿌려주는 과정
fig, ax = plt.subplots(1, figsize=(20,20))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors): # descriptors만큼 루프를 돌고
    found = False
    for name, saved_desc in descs.items(): # 얼굴 인코드된 값을 저장한 descs에 루프 돌면서 // 누가 누군지 비교하는 부분
        dist = np.linalg.norm([desc]-saved_desc, axis=1) # distance linear algebra norm  유클리디안 distance 거리 구함/
        
        if dist<0.6: # 0.6 이하일 때 성능이 제일 좋아
            found = True
            # 그리는 부분
            text = ax.text(rects[i][0][0], rects[i][0][1],name, # 찾으면 그 사람의 이름을 이름 써라
                           color='b', fontsize=40, fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'),
            path_effects.Normal()])
            
            rect = patches.Rectangle(rects[i][0], # 얼굴 부분을 사각형으로 그림 
                                     rects[i][1][1] - rects[i][0][1],
                                     rects[i][1][0] - rects[i][0][0],
                                     linewidth=2, edgecolor='w', facecolor='none'
                                     )
            ax.add_patch(rect)

            break
    if not found: # 못찾으면 unknown으로 표시
        ax.text(rects[i][0][0], rects[i][0][1], 'unknown',color='r', fontsize=20, fontweight='bold')
        rect = patches.Rectangle(rects[i][0],
                                 rects[i][1][1] - rects [i][0][1],
                                 rects[i][1][0] - rects [i][0][0],
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.axis('off')
plt.savefig('teamproject/simple_face_recognition-master/img/output6.png')
plt.show()
        