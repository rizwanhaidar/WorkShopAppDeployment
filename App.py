from flask import Flask, request, jsonify, render_template
import joblib
from pyforest import *
#naming our app as app
app= Flask(__name__)

#loading the pickle file for creating the web app
model= joblib.load(open("model.pkl", "rb"))

#defining the different pages of html and specifying the features required to be filled in the html form
@app.route("/")
def home():
    return render_template("index.html")

#creating a function for the prediction model by specifying the parameters and feeding it to the ML model
@app.route("/predict", methods=["POST"])
def predict():
    #specifying our parameters as data type float
    int_features= [float(x) for x in request.form.values()]
    final_features= [np.array(int_features)]
    prediction= model.predict(final_features)
    output= round(prediction[0], 2)
    return render_template("index.html", prediction_text= "flower is {}".format(output))

#running the flask app
if __name__ == '__main__':
    app.run(debug=True)