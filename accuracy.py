import matplotlib.pyplot as plt

metrics = {
    "Accuracy": 0.9865,
    "Precision": 0.9781,
    "F1 Score": 0.9865
}

plt.figure(figsize=(6,4))
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Fake News Detection Model Performance")
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Fake", "True"])

disp.plot()
plt.title("Confusion Matrix")
plt.show()