import streamlit as st
st.title("Iris Classifier- API")

sl = st.slider('Sepal Length', 4.3, 7.9, 0.5)
sw = st.slider('Sepal Width', 2.0, 4.4, 0.5)
pl = st.slider('Petal Length', 1.0, 6.9, 0.5)
pw = st.slider('Petal Width', 0.1,2.5, 0.5)

from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(iris.data,iris.target)

op = model.predict([[sl,sw,pl,pw]])
op = iris.target_names[op[0]]
st.title(op)
if(op=="setosa"):
   st.image("https://images.unsplash.com/photo-1561831220-cc44b32786ca?ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTd8fGlyaXMlMjBmbG93ZXJ8ZW58MHx8MHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60")
elif(op=="versicolor"):
    st.image("https://images.unsplash.com/photo-1602008903362-c8647f986df5?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=750&q=80")
else:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkeb-oFhav9r_dcv8TmNFJ9JZVB-6iKJyYXA&usqp=CAU")
