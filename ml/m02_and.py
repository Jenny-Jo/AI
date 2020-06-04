from sklearn.svm import LinearSVC #회귀 모델
from sklearn.metrics import accuracy_score

# 1. Data
x_data = [[0,0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 0, 0, 1]

# 2. Model
model = LinearSVC()
# 이거 하나면 끝

# 3. Fit
model.fit(x_data, y_data)

# 4. Evaluate = score, Predict
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)

acc = accuracy_score([0, 0, 0, 1], y_predict)


print(x_test, "의 예측 결과: ", y_predict)
print("acc = ", acc)