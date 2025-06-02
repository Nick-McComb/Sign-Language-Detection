import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np


data_dict = pickle.load(open('data.pickle', 'rb'))

#print(data_dict['data'])
#print(data_dict['labels'])


data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2, shuffle= True, stratify= labels)

model = RandomForestClassifier()

model.fit(train_x, train_y)

predicted_y = model.predict(test_x)

accuracy = accuracy_score(predicted_y, test_y)

print("The accuracy of this model is {}%" .format(accuracy*100))

f = open('model.p', 'wb')
pickle.dump({"model": model}, f)
f.close()





