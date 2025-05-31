import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

train_df = pd.read_csv('/Users/vbajoria/desktop/titanic-train.csv')
print(train_df.head(5))

def manipluate_df(df):

    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df['Age'].fillna(value = df['Age'].mean(), inplace= True)
    df['FirstClass'] = df['Pclass'].map(lambda x: 1 if  x== 1 else 0)
    df['SecondClass'] = df['Pclass'].map(lambda x: 1 if x == 2 else 0)
    df['ThirdClass'] = df['Pclass'].map(lambda x: 1 if x == 3 else 0)

    df = df[['Sex' , 'Age' , 'FirstClass', 'SecondClass' ,'ThirdClass' , 'Survived'] ]

    return df

manipulated_df= manipluate_df(train_df)
print(manipulated_df)

features = manipulated_df[['Sex' , 'Age' , 'FirstClass', 'SecondClass' ,'ThirdClass']]
survival = manipulated_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(features, survival, test_size=0.3, random_state=42)


#Standard scalar standardizes by removing the mean and scaling to unit variance, we frst find mean and std on the training data and then
#we use the same mean and std on test data too

#Fitting the scaler to X_test would compute new mean and standard deviation values based on the test data,
# which introduces bias and violates the principle of treating test data as unseen.

scaler = StandardScaler()

train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

#create the model

model = LogisticRegression()
model.fit(train_features, y_train)

train_score = model.score(train_features, y_train)
test_score = model.score(test_features, y_test)

y_predict = model.predict(test_features)


print(train_score)
print(test_score)

st.title("Would you have survived the Titanic disaster ?")
st.subheader("This model predicts if he/she survives the titanic disaster")
st.dataframe(train_df.head())

confusion = confusion_matrix(y_test, y_predict)

FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]

st.subheader("Training set scores: {}".format(round(train_score, 3)))
st.subheader("Testing set scores: {}".format(round(test_score, 3)))

plt.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP])

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP])
ax.set_xlabel('Confusion matrix')
st.pyplot(fig)


















