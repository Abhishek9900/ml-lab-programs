from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt


# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target
# print(y)

# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create one-vs-rest logistic regression object
clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')

# Train model
model = clf.fit(X_std, y)

# Create new observation
new_observation = [[.5, .5, .5, .5]]

# Predict class
print(model.predict(new_observation))

# View predicted probabilities
print(model.predict_proba(new_observation))

# Confusion Matrix
y_target = [1, 1, 1, 0, 0, 2, 0]
y_predicted = [1, 0, 1, 0, 0, 2, 1]

cm = confusion_matrix(y_target=y_target,
                      y_predicted=y_predicted,
                      binary=False)
# print(cm)

fig, ax = plot_confusion_matrix(conf_mat=cm)


from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


y = label_binarize(y, classes=[0,1,2])
n_classes = 3

# shuffle and split training and test sets
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.33, random_state=0)

# classifier
clf_auc = OneVsRestClassifier(LinearSVC(random_state=0))
y_score = clf_auc.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
f, axarr = plt.subplots(3, sharex=True, sharey=True)
f.suptitle('Receiver Operating Characteristic')
print(type(axarr))
for ax in axarr.flat:
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')

for i in range(n_classes):
    axarr[i].plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    axarr[i].plot([0, 1], [0, 1], 'k--')
    axarr[i].set_xlim([0.0, 1.0])
    axarr[i].set_ylim([0.0, 1.05])
    # axarr[i].set_xlabel('False Positive Rate')
    # axarr[i].set_ylabel('True Positive Rate')
    axarr[i].legend(loc="lower right")

# Bring subplots close to each other.
f.subplots_adjust(hspace=0)
# Hide x labels and tick labels for all but bottom plot.
for ax in axarr.flat:
    ax.label_outer()
plt.show()

