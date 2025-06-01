import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

train_df = pd.read_csv('./data/titanic-train.csv')
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



def manipulate_user_input( sex, p_class):
    if sex == 'Male':
        sex =0
    else:
        sex =1

    f_class = 1 if p_class == 'First Class' else 0
    s_class = 1 if p_class == 'Second Class' else 0
    t_class = 1 if p_class == 'Third Class' else 0
    return f_class, s_class, t_class, sex


def predict_passenger_survival(f,s,t,e, a):

    input_data = scaler.transform([[e,a, f,s,t]])
    prediction = model.predict(input_data)

    predict_probability = model.predict_proba(input_data)

    return prediction, predict_probability


def get_user_input():
    name = st.text_input("Enter the passenger name")
    sex = st.selectbox("choose gender", options=["Male", "Female"])
    age = st.slider("Select age", 1, 100, 1)
    p_class = st.selectbox("Passenger class", options=['First Class', 'Second Class', 'Third Class'])
    print(p_class)

    f_class, s_class, t_class, sex = manipulate_user_input(sex, p_class)

    prediction, pred_prob = predict_passenger_survival(f_class, s_class, t_class, sex, age)

    if prediction[0] == 1:
        st.subheader("Passenger {} would have survived with probability {}".format(name, round(pred_prob[0][1]*100 , 3)))
    else:
        st.subheader("Passenger {} would not have survived with probability {}".format(name, round(pred_prob[0][1]*100 , 3)))

get_user_input()

























