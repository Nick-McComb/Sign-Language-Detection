import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data_dict = pickle.load(open('data.pickle', 'rb'))

#print(data_dict['data'])
#print(data_dict['labels'])




