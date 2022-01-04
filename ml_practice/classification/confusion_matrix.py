# Classify can_decide and cb results

import csv
import  matplotlib.pyplot as plt
from sklearn import metrics

can_decide_filename = "can_decide_results.csv"
cb_filename = "cb_results.csv"

# The accuracy of the model
def accuracy_model(tp, tn, fp, fn):
    return (tn + tp) / (tp + tn + fp + fn) * 100

# Miscalculation rate
def classification_error(tp, tn, fp, fn):
    return (fn + fp) / (tp + tn + fp + fn)  *100

# Sensitivity / True positive rate
# Tells us what % correctly detected.
def recall(tp, tn, fp, fn):
    return tp / (tp + fn) * 100

# Specificity
# Tells us what % correctly rejected.
def specificity(tp, tn, fp, fn):
    return tn / (tn + fp) * 100

# Positive predicted value
# The ratio of correct positive predictions to the total predicted positives.
def precision(tp, tn, fp, fn):
    return tp / (tp + fp) * 100

# F1 Score
def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def plot_confusion_matrix(cm, title="Confusion matrix", cmap=plt.cm.Blues): 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class CsvRead():
    predictions = []
    actuals = []

    tp_data = []
    tn_data = []
    fp_data = []
    fn_data = []
    
    def __init__(self, filename):
        self.filename = filename

    def classify_result(self):
        # clear the lists!
        self.predictions.clear()
        self.actuals.clear()
        self.tp_data.clear()
        self.tn_data.clear()
        self.fp_data.clear()
        self.fn_data.clear()

        with open(self.filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                prediction = int(row[3])
                actual = int(row[4])
                self.predictions.append(prediction)
                self.actuals.append(actual)
                if (prediction == 1): 
                    if (actual == 1): 
                        self.tp_data.append(row)
                    else:
                        self.fn_data.append(row)
                else: 
                    if (actual == 1):
                        self.fp_data.append(row)
                    else:
                        self.tn_data.append(row)
        
    # You can also use sklearn library to calculate f1_score / recall etc...
    def show_cm_result(self, title):
        cm = metrics.confusion_matrix(self.actuals, self.predictions)
        print(cm)
        plt.figure()
        plot_confusion_matrix(cm, title)
        plt.show()

    def show_result(self, title):
        tp = len(self.tp_data)
        tn = len(self.tn_data)
        fp = len(self.fp_data)
        fn = len(self.fn_data)
        
        print("\nTP: ", tp, ", TN: ", tn, " , FP: ", fp, ", FN: ", fn)

        print("\n=== Detailed Accuracy ===\n")
        print("\nAccuracy: ", accuracy_model(tp, tn, fp, fn))
        print("\nClassification Error: ", classification_error(tp, tn, fp, fn))

        recall_result = recall(tp, tn, fp, fn)
        precision_result = precision(tp, tn, fp, fn)
        print("\nRecall / Sensitivity: ", recall_result)
        print("\nSpecificity: ", specificity(fp, tn, fp, fn))
        print("\nPrecision: ", precision_result)
        print("\nF1 Score: ", f1_score(precision_result, recall_result))
        
        print("\n=== Confusion Matrix ===\n")
        self.show_cm_result(title)

        print("\n=== Classification Report ===\n")
        class_report = metrics.classification_report(self.actuals, self.predictions, labels=[1, 0])
        print("Classification Report : \n", class_report)


csvRead_can_decide = CsvRead(can_decide_filename)
csvRead_can_decide.classify_result()
print("=== Can Decide Result Summary ===\n")
csvRead_can_decide.show_result(title="Can Decide Confusion matrix")

csvRead_cb = CsvRead(cb_filename)
csvRead_cb.classify_result()
print("=== Cumulonimbus Result Summary ===\n")
csvRead_cb.show_result(title="Cb Confusion matrix")