import cv2
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def read_data(fin):
    target_li = []
    data_li = []
    for line in open(fin):
        image_path, face_id = line.strip().split(';')
        image_data = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
        data_li.append(image_data)
        target_li.append(int(face_id))
    return (np.array(data_li), np.array(target_li))


def create_train_test_data(image_data, label_li):
    n_samples, imgae_h, image_w = image_data.shape
    x = image_data.reshape(n_samples, -1)
    n_features = x.shape[1]
    y = label_li
    
    n_classes = len(set(y))
    print('Total dataset size:')
    print('n_samples: %d' % n_samples)
    print('n_features:%d'%n_features)
    print('n_classes:%d'%n_classes)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=42)
    return(x_train, x_test, y_train, y_test)

def extract_features(x_train, x_test, n_components):
    print('Extracting the top %d eigenfaces from %d faces'%(n_components, x_train.shape[0]))
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    return(x_train_pca, x_teset_pca)

def train_test_classifier(x_train_pca, x_test_pca, y_train, y_test):
    print('Fitting the classifier to the training set')
    param_grid = {'C':[1e3, 5e3, 1e4, 5e4, 1e5], 'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.01]}
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(x_train_pca, y_train)
    print(clf.best_estimator_)

    print("predicting people's names on the test set")
    y_pred = clf.predict(x_test_pca)
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    argv = sys.argv
    image_dta, label = read_data('faces.csv')
    n_eigenface = 10
    x_train, x_test, y_train, y_test, = create_train_test_data(image_data, label)
    x_train_pca, x_test_pca = extract_features(x_train, x_test, n_eigenface)
    train_test_classifier(x_train_pca, x_test_pca, y_train, y_test)

    