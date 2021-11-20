import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selectiom import train_test_split
from sklearn.ensemble import RnaomForestClassifier
from PIL import Image
import streamlit as st


#Criacao de um título e subtitulo
st.write("""
#Delirium Detection
Detect if someone has delirium using maachine learning and python !
""")

#Abrir e mostrar uma imagem

image = Image.open('/Users/user/PycharmProjects/WebApp/delirium.jpeg')
st.image(image, caption='ML', use_column_with=True)

#Dados

df = pd.read_csv('/Users/user/PycharmProjects/WebApp/diabetes.csv')

#configurar um sub titulo
st.subheader('Data Information:')

#Mostrar os datods como um dataframe/tabela
st.dataframe(df)

st.write(df.describe())

chart = st.bar_chart(df)

#dividir os dados en independentes 'x' e dependentes 'y'

X = df.iloc[:,0:8].values
Y = df.iloc[:,-1].values

#dividir os dados em 75% para treino e 25% para teste
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25, random_state=0)

#Buscar o input do utilizador

def get_user_input ():
    pregnancies = st.sidebar.slider('pregnacies',0,17,3)
    glucose = st.sidebar.slider('glucose', 0,199,117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0,122,72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0,99,23)
    insulin = st.sidebar.slider('insulin', 0.0,846.0,30.5)
    BMI = st.sidebar.slider('BMI', 0.0,67.1,32.0)
    DPF = st.sidebar.slider('DPF', 0.078,2.42,0.3725)
    age = st.sidebar.slider('age', 21,81,29)

    #Guardar o dicionário numa variável

    user_data = {'pregnancies': pregnacies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'age': age
                 }

    #Transformar os dados num dataframe

    features = pd.DataFrame(user_data, index= [0])
    return features

#guardar o input do utilizador numa variavel

user_input = get_user_input()

#Configurar uma subhead e mostrar aos utilizadores input
st.subheader('User Input:')
st.write(user_input)

#criar e treinar o modelo

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

#Mostar as métricas do modelo
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

#Guardar o modelos preditos numa variavel
prediction = RandomForestClassifier.predict(user_input)

#configurar um subheader e mostrar a classificação
st.subheader('Classification:')
st.write(prediction)
