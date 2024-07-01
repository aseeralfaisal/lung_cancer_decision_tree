import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('./dataset/dataset.csv')
data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 2})

X = data[['AGE', 'SMOKING', 'ANXIETY', 'ALCOHOL_CONSUMING', 'GENDER']]
Y = data['LUNG_CANCER']

model = DecisionTreeClassifier()
model.fit(X.values, Y)
prediction = model.predict(
    np.vstack([[[20, 2, 2, 2, 2]], [[20, 1, 1, 1, 2]], [[20, 1, 2, 1, 2]]]))

print(data, prediction)
