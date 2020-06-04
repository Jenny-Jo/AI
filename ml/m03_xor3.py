from sklearn.svm import LinearSVC, SVC #회귀 모델
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# 특정 모델빼곤 legacy machine learning 에선 잘 쓰지 않는다. 취업해서 나중에 훑어보기


# 1. Data
x_data = [[0,0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0]

# 2. Model
# model = LinearSVC()
# model = SVC()
model = KNeighborsClassifier(n_neighbors=1)



# 이거 하나면 끝

# 3. Fit
model.fit(x_data, y_data)

# 4. score, Predict
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)

acc = accuracy_score([0, 1, 1, 0], y_predict)


print(x_test, "의 예측 결과: ", y_predict)
print("acc = ", acc)