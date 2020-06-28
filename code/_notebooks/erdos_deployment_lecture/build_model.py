
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


# load dataset
iris = datasets.load_iris()
df_iris = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
df_iris['target'] = iris['target']

# create train test split
df_train, df_test = train_test_split(df_iris,
                                     test_size=0.3,
                                     random_state=47,
                                     stratify=df_iris['target'])

# put together pipeline
model = Pipeline([
    ('scale', StandardScaler())
    ,('gbm', GradientBoostingClassifier())
])

# make model
model.fit(df_train[iris['feature_names']], df_train['target'])

# see how accurate it is on test data
test_predictions = model.predict(df_test[iris['feature_names']])
test_accuracy = accuracy_score(df_test['target'], test_predictions)
print('Test accuracy of fit model: {:.3f}'.format(test_accuracy))

# save model
pickle.dump(model, open('./model.pickle', 'wb'))
