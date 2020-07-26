import pandas as pd
df = pd.read_csv('daily_weather.csv')
df.head()
df.info()
df[df.isnull().any(axis=1)]
df.drop('number',axis=1,inplace=True)
df.dropna(inplace=True)
df.info()
df.head()
df2= df.copy()
df2['humidityval']=(df2['relative_humidity_3pm']>25)*1
df2.head()
df2.drop('relative_humidity_3pm',axis=1,inplace=True)
y=df2['humidityval']
x=df2.drop('humidityval',axis=1)
x.head()
from sklearn.model_selection import train_test_split
trainx, testx, trainy, testy= train_test_split(x,y,test_size=0.33)
from sklearn.tree import DecisionTreeClassifier
dtcmodel = DecisionTreeClassifier()
dtcmodel.fit(trainx, trainy)
predictedop = dtcmodel.predict(testx)



from sklearn.metrics import accuracy_score
accuracy_score(testy,predictedop)
from sklearn.metrics import confusion_matrix
confusion_matrix(testy,predictedop)
from sklearn.ensemble import RandomForestClassifier
rfcmodel = RandomForestClassifier(n_estimators=10)
rfcmodel.fit(trainx,trainy)
rfcprediction=rfcmodel.predict(testx)
accuracy_score(rfcprediction,testy)
confusion_matrix(rfcprediction,testy)