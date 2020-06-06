import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris.csv', header=0)

x = iris.iloc[:,0:4]
y = iris.iloc[:, 4]
print(x)
print(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=44, shuffle = True) #shuffle true 가 default

kfold=KFold(n_splits=5, shuffle=True)






allAlgorithms = all_estimators(type_filter = 'classifier') # 26개 모델

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    scores = cross_val_score(model, x, y, cv= kfold) # acc
    print(name, '의 정답률 = ')
    print(scores)
    # model.fit(x, y)

    y_pred = model.predict(x)
    print(name, "의 정답률 =", accuracy_score(y, y_pred))

# 각 모델별로 5번씩 돌린 acc 가 나옴
import sklearn
print(sklearn.__version__)
'''
모델들의 정답률

AdaBoostClassifier 의 정답률 = 0.9666666666666667
BaggingClassifier 의 정답률 = 0.9666666666666667
BernoulliNB 의 정답률 = 0.3
CalibratedClassifierCV 의 정답률 = 0.9666666666666667
ComplementNB 의 정답률 = 0.7
DecisionTreeClassifier 의 정답률 = 0.8666666666666667
ExtraTreeClassifier 의 정답률 = 0.8666666666666667
ExtraTreesClassifier 의 정답률 = 0.9666666666666667
GaussianNB 의 정답률 = 0.9333333333333333
GaussianProcessClassifier 의 정답률 = 0.9666666666666667
GradientBoostingClassifier 의 정답률 = 0.9333333333333333
KNeighborsClassifier 의 정답률 = 0.9666666666666667
LabelPropagation 의 정답률 = 0.9666666666666667
LabelSpreading 의 정답률 = 0.9666666666666667
LinearDiscriminantAnalysis 의 정답률 = 1.0
LinearSVC 의 정답률 = 0.9666666666666667
LogisticRegression 의 정답률 = 1.0 분류!!
LogisticRegressionCV 의 정답률 = 0.9
MLPClassifier 의 정답률 = 1.0
MultinomialNB 의 정답률 = 0.8666666666666667
NearestCentroid 의 정답률 = 0.9
NuSVC 의 정답률 = 0.9666666666666667
PassiveAggressiveClassifier 의 정답률 = 0.6
Perceptron 의 정답률 = 0.5333333333333333
QuadraticDiscriminantAnalysis 의 정답률 = 1.0
RadiusNeighborsClassifier 의 정답률 = 0.9333333333333333
RandomForestClassifier 의 정답률 = 0.9666666666666667
RidgeClassifier 의 정답률 = 0.8333333333333334
RidgeClassifierCV 의 정답률 = 0.8333333333333334
SGDClassifier 의 정답률 = 0.7
SVC 의 정답률 = 0.9666666666666667

sklearn 0.20.1에서 제공하는 모든 모델이다
'''