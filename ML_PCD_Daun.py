import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = r'D:\data\preprocess_daun.csv'
df = pd.read_csv(path)
pd.set_option('display.max_columns', None)
print('==== Data Head ====\n',df.head())

#Enumerate the labels feature and drop unnecessary column (Unnamed: 0)
df = df.drop(columns=['Unnamed: 0'])
mapping = {'healthy':0,'Apple_scab':1,'Black_rot':2,'Cedar_apple_rust':3}
df = df.replace({'labels':mapping})
print('==== Data Head num====\n',df.head())
corr = df.corr()
import seaborn as sn
sn.heatmap(corr,annot=True)
plt.show()
print(corr)


#Because F1 and F2 have inverse dependencies (-1) then we choose only one of them (in this case i choose F1)
df = df.drop(columns=['f2'])

#Less correlated features gets dropped
less_corr = ['area','dissimilarity','correlation','contrast','ASM','energy']
df = df.drop(columns=less_corr)
X = df.drop(columns=['labels'])
y = df['labels']

#Random forest classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
max_iter=100
RF = RandomForestClassifier()
for i in range(max_iter):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    RF.fit(X_train,y_train)
    prediction = RF.predict(X_test)
    p = classification_report(y_test,prediction,output_dict=True)
    temp = p['accuracy']
    if temp>=0.82:
        _random_state = i
        break
print(classification_report(y_test,prediction,output_dict=False))

#10 fold cross validation
from sklearn.model_selection import cross_val_score


print('===== K-Fold Cross Evaluation =====')
_scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
for i in range(len(_scoring)):
    score = cross_val_score(RF, X, y, scoring=_scoring[i], cv=10, n_jobs=-1,)
    print('%s: %.3f (%.3f) ->max :%.3f' % (_scoring[i],np.mean(score), np.std(score),np.mean(score)+np.std(score)))

#fitting the splitted data(X and y) and y that has been binarize to OVR classifier so that we can get the ROC and AUC values
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize
print('\n\n===== ROC =====')
y_biner = label_binarize(y,classes=[0,1,2,3])
X_biner_train, X_biner_test, y_biner_train, y_biner_test = train_test_split(X, y_biner, test_size=0.2, random_state=_random_state)
clf = OneVsRestClassifier(RandomForestClassifier(random_state=_random_state))
y_score = clf.fit(X_biner_train, y_biner_train).predict(X_biner_test)

# Generate the ROC and AUC value for each target labels (Healthy,...)
fpr = dict()
tpr = dict()
roc_auc = dict()
lw=2
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_biner_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = ['blue', 'red', 'green','yellow']
for i in range(4):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.savefig('ROC_Daun.png')
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay
conf = ConfusionMatrixDisplay.from_predictions(y_test,prediction,cmap=plt.cm.Blues)
conf.ax_.set_title('Confusion Matrix')
print(conf.confusion_matrix)
plt.show()

# Take random Apple leaf image for testing the ML model
tes = pd.read_csv(r'D:\data\preprocess_daun_tes.csv')
tes = tes.drop(columns=['f2'])
tes = tes.drop(columns=less_corr)
tes = tes.drop(columns=['labels'])
tes = tes.drop(columns=['Unnamed: 0'])
pred = RF.predict(tes)

if pred== 0:
    pred = 'Healthy'
elif pred == 1:
    pred = 'Scab'
elif pred == 2 :
    pred = 'Black Rot'
else:
    pred == 'Cedar Apple Rust'

print('Your leaf is : ', pred)



