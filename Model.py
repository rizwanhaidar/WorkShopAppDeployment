from pyforest import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.datasets import load_iris
#loading the dataset into a pandas dataframe
data= load_iris()
data.feature_names
df= pd.DataFrame(data.data)
df.head()
#renaming the columns with the actual column names that is sepal and petal width and length
df.columns= data.feature_names
data.target_names
#inserting the target feature in the dataset
df["target"]= data.target
#getting our X and y to feed it into the ML model
X= df.drop("target", axis=1)
y= df.target
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=42)

model= SVC()
#training the model
model.fit(X_train,y_train)
#making predictions
y_pred= model.predict(X_test)
#pickling the model
joblib.dump(model, "model.pkl")
c= [2,3,3,4]
from_jb= joblib.load("model.pkl")
from_jb.predict([c])