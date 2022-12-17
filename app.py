import streamlit as st
st.title("AI to find Iris species")
st.subheader("Enter the Iris's properties and find out it's species")
st.subheader("MADE BY RIBENCE KADEL")
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from PIL import Image




a=(st.number_input(label_visibility="visible",label="Enter sepal length of an Iris (in cms)"))
b=st.number_input(label_visibility="visible",label="Enter sepal width of an Iris (in cms)")
c=st.number_input(label_visibility="visible",label="Enter petal length of an Iris (in cms)")
d=st.number_input(label_visibility="visible",label="Enter petal width of an Iris (in cms)")

# st.write(type(box_a))

knn=KNeighborsClassifier(n_neighbors=1)
iris_dataset= load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

X_new=np.array([[a,b,c,d ]])
knn.fit(X_train,y_train)

# print("Prediction: {}".format(prediction))
# print("Predicted species: {}".format(iris_dataset['target_names'][prediction]))



location=st.container()

y_pred= knn.predict(X_test)

smth=knn.score(X_test,y_test)
accuracy=smth*100

def qwerty():
    prediction=knn.predict(X_new)
    # st.write("Predicted species: {}".format(iris_dataset['target_names'][prediction]))
    
    if (prediction == 0):
        st.write("The predicted species is Iris Setosa")
        st.write("Family: Iridaceae,  Order: Asparagales,           Kingdom: Plantae")
        st.write("The accuracy of the Prediction is: {}%".format(accuracy) )
        
        image= Image.open('iris_setosa.jpg')
        st.image(image,caption="IRIS SETOSA")
    elif (prediction == 1):
        st.write("The predicted species is Iris Versicolor")
        st.write("Higher classification: Irises,Family: Iridaceae,Order: Asparagales")
        st.write("The accuracy of the Prediction is: {}%".format(accuracy ) )        
        image1=Image.open('Iris-Cat-Mousam.jpg')
        st.image(image1,caption="IRIS VERSCOLOR")
    else:
        st.write("The predicted species is Iris Verginica")
        st.write("Scientific name: Iris virginica, Higher classification: Irises, Rank: Species")
        st.write("The accuracy of the Prediction is: {}%".format(accuracy) )
        image3=Image.open('Iris_virginica.jpg')
        st.image(image3,caption="IRUS VERGINICA")


st.button(label="Find the Species",on_click=qwerty)
