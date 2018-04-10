#import the classifier library
from sklearn import tree
#describe the feature dataset (format: weight in gms, 0(bumpy) or 1(smooth) for texture)
features = [[140,1],[130,1], [150,0], [170,0]]
#describe the labels for the feature set (orange is 1, apple is 0)
labels = [1,1,0,0]
#print ("the labels are", labels)
#define a variable to use the DecisionTree classifer
clf = tree.DecisionTreeClassifier()
#pass the dataset into the classifier using the in-built fit module
clf = clf.fit(features,labels)
#print the result of our test dataset
print ("The image is a",clf.predict([[160,0]]))
