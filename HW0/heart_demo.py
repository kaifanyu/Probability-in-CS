'''
    UCI CS177: Heart Disease Prediction
    This is DEMONSTRATION code:
    - It gives an example of how to load the heart disease training & test data,
      and use this data to compute simple probabilities.
    - You may (but do not have to) reuse parts of this code in your solutions. 
    - It is NOT a template for the individual questions you must answer,
      for that see the main homework pdf.
'''
 
import numpy as np

trainPatient = np.load('trainPatient.npy')
testPatient = np.load('testPatient.npy')
''' 
    Data description:
    - trainPatient has one row per training example
    - testPatient has one row per test example
    - column 1 is age feature (in years), column 2 is exercise feature (binary)
    - column 3 is true category label (1 for Disease, 0 for Healthy)
    - Data source: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
'''

# Estimate the probability of heart disease
N = trainPatient.shape[0]
Ndisease = np.sum(trainPatient[:, 2])
Pdisease = Ndisease / N
print('Probability of heart disease:', Pdisease)

# Find accuracy of simple baseline classifier that predicts all patients
# are healthy (the more common of the two categories)
M = testPatient.shape[0]
Yhat = np.zeros(M)
accuracy = np.sum(testPatient[:, 2] == Yhat) / M
print('Accuracy of baseline classifier:', accuracy)
