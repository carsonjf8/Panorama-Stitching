# Usage: python ./main.py --train ./train --test ./test
# Usage: python ./main.py --train ./train --test ./train

import os
import cv2
import getopt
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['train=', 'test='])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    train_fn = ''
    test_fn = ''

    for o, a in opts:
        if o == '--train':
            train_fn = a
        elif o == '--test':
            test_fn = a
        else:
            assert False, "unhandled option"

    tm, te, tp = eigenfaces(train_fn)
    recog_clf(test_fn, tm, te, tp)

def eigenfaces(train_fn):
    # create training image matrix
    train_mx = create_img_mx(train_fn)
    #print('train_mx.shape', train_mx.shape)
    train_mean = np.mean(train_mx, axis=0)
    #print('train_mean.shape', train_mean.shape)
    train_dif = train_mx - train_mean
    #print('train_dif.shape', train_dif.shape)

    # get eigenvectors
    train_eigval, train_eigvec = np.linalg.eig(train_dif @ train_dif.T)
    #print('train_eigval.shape', train_eigval.shape)
    #print('train_eigvec.shape', train_eigvec.shape)
    sorted_indices = np.flip(train_eigval.argsort())
    sorted_train_eigval = train_eigval[sorted_indices]
    sorted_train_eigvec = train_eigvec[:, sorted_indices]
    #print('sorted_train_eigval.shape', sorted_train_eigval.shape)
    #print('sorted_train_eigvec.shape', sorted_train_eigvec.shape)
    eigenvec = (train_dif.T @ sorted_train_eigvec.T).T
    #print('eigenvec.shape', eigenvec.shape)

    # project into subspace
    train_prods = []
    for train in train_dif:
        train_prods.append(np.dot(eigenvec, train.T).T)
    train_prods = np.array(train_prods)
    #print('train_prods.shape', train_prods.shape)
    #print()
    return train_mean, eigenvec, train_prods

def recog_clf(test_fn, train_mean, train_eigvecs, train_prods):
    # create testing image matrix
    test_mx = create_img_mx(test_fn)
    #print('test_mx.shape', test_mx.shape)
    test_dif = test_mx - train_mean
    #print('test_dif.shape', test_dif.shape)

    # project into subspace
    test_prods = []
    for test in test_dif:
        test_prods.append(np.dot(train_eigvecs, test.T).T)
    test_prods = np.array(test_prods)
    #print('test_prods.shape', test_prods.shape)

    # recognize faces
    predictions = np.full(test_mx.shape[0], -1, dtype=int)
    for test_index, test_img in enumerate(test_prods):
        closest_dist = 999999999999999999
        for train_index, train_img in enumerate(train_prods):
            #dist = np.dot(test_img, train_img)
            dist = calc_dist(test_img, train_img)
            if dist < closest_dist:
                closest_dist = dist
                predictions[test_index] = train_index
    print('predictions', predictions)
    accuracy = 0
    for index, pred in enumerate(predictions):
        if pred == index:
            accuracy += 1
    accuracy /= test_mx.shape[0]
    print('accuracy', accuracy)

def calc_dist(a, b):
    total = 0
    for (c, d) in zip(a, b):
        total += pow(c - d, 2)
    total = pow(total, 0.5)
    return total

def create_img_mx(folder_fn):
    file_list = os.listdir(folder_fn)
    img = cv2.imread(folder_fn + '/' + file_list[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_mx = np.zeros((len(file_list), gray.shape[0] * gray.shape[1]))
    for index, fn in enumerate(file_list):
        img = cv2.imread(folder_fn + '/' + fn)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat = gray.flatten()
        img_mx[index] = flat
    return img_mx

if __name__ == "__main__":
    main()
