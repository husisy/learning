import numpy as np
from sklearn import datasets, svm

digits = datasets.load_digits()
ind1 = np.random.permutation(digits.target.shape[0])
data = digits.data[ind1]
label = digits.target[ind1]
clf = svm.SVC(C=1, kernel='linear')

data_fold = np.array_split(data, 3)
label_fold = np.array_split(label, 3)
for ind1 in range(len(data_fold)):
    tmp1 = list(data_fold) #use 'list' to copy
    data_val  = tmp1.pop(ind1)
    data_train = np.concatenate(tmp1)
    
    tmp1 = list(label_fold)
    label_val  = tmp1.pop(ind1)
    label_train = np.concatenate(tmp1)
    
    clf.fit(data_train, label_train)
    print(ind1, clf.score(data_val, label_val))
