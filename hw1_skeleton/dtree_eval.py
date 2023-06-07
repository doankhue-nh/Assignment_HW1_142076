'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

numOfFoldsPerTrial = 10

def evaluatePerformance(numTrials=100):
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = r'D:\Assignment\CIS419\Assignment1\hw1_skeleton\data\SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',') # Data: mảng 267 hàng 45 cột 
    X = data[:, 1:] # Tất cả hàng cột trừ cột đầu tiên => mảng 267 hàng 44 cột
    y = np.array([data[:, 0]]).T # Cột đầu tiên transpose => mảng 1 cột chứa label
    n,d = X.shape

    # create list to hold data
    treeAccuracies = []
    stumpAccuracies = []
    dt3Accuracies = []

    # perform 100 trials
    for x in range(0, numTrials):
        print(x)
        # shuffle the data
        idx = np.arange(n) # Khoi tao mang idx so nguyen co gia tri tu 1 toi 266
        np.random.seed(13) # Khoi tao mang ngau nhien
        np.random.shuffle(idx) # Trộn lại mảng idx, các phần tử nằm vị trí ngẫu nhiên
        X = X[idx] # Phương thức ngẫu nhiên hóa các phần tử của mảng X dựa vào vị trí chỉ số mảng idx cung cấp
        y = y[idx] # Tương tự trên

        # split the data randomly into 10 folds
        folds = []    
        intervalDivider = len(X)/numOfFoldsPerTrial
        for fold in range(0, numOfFoldsPerTrial):
            # designate a new testing range
            #X[Duyệt phần tử theo hàng : Duyệt phần tử theo cột ]
            Xtest = X[int(fold * intervalDivider):int((fold + 1) * intervalDivider),:] #Ví dụ fold bằng 0, lấy hàng có index 0
            ytest = y[int(fold * intervalDivider):int((fold + 1) * intervalDivider),:]
            Xtrain = X[:int((fold * intervalDivider)),:]
            ytrain = y[:int((fold * intervalDivider)),:]
            Xtrain = Xtrain.tolist()
            ytrain = ytrain.tolist()
            # Với Xtrain, tại index = 0 list rỗng, index = 1, list có 1 phần từ,...
            # Append các phần tử còn lại (Trừ Xtest, ytest)
            for dataRow in range(int((fold + 1) * intervalDivider), len(X)):
                Xtrain.append(X[dataRow])
                ytrain.append(y[dataRow])

            # Train decision tree
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(Xtrain,ytrain)
            # output predictions on the remaining data
            y_pred_tree = clf.predict(Xtest)

            # train the 1-level decision tree
            oneLevel = tree.DecisionTreeClassifier(max_depth=1)
            oneLevel = oneLevel.fit(Xtrain,ytrain)
            # output predictions on the remaining data
            y_pred_stump = oneLevel.predict(Xtest)

            # train the 3-level decision tree
            threeLevel = tree.DecisionTreeClassifier(max_depth=3)
            threeLevel = threeLevel.fit(Xtrain,ytrain)
            # output predictions on the remaining data
            y_pred_dt3 = threeLevel.predict(Xtest)

            # Tính toán training accuracy của model và save vào list chứa tất cả độ chính xác
            treeAccuracies.append(accuracy_score(ytest, y_pred_tree))
            stumpAccuracies.append(accuracy_score(ytest, y_pred_stump))
            dt3Accuracies.append(accuracy_score(ytest, y_pred_dt3))
        #Hàm accuracy_score: In multilabel classification, this function computes subset accuracy:
        # the set of labels predicted for a sample must *exactly* match the
        # corresponding set of labels in y_true. 
    
    # Tính toán mean và std của các accuracy thu được
    meanDecisionTreeAccuracy = np.mean(treeAccuracies)
    stddevDecisionTreeAccuracy = np.std(treeAccuracies)
    meanDecisionStumpAccuracy = np.mean(stumpAccuracies)
    stddevDecisionStumpAccuracy = np.std(stumpAccuracies)
    meanDT3Accuracy = np.mean(dt3Accuracies)
    stddevDT3Accuracy = np.std(dt3Accuracies)

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print ("Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")")
    print ("Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")")
    print ("3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")")
# ...to HERE.