from flask import Flask, render_template,request
import pickle
import numpy as np

model = pickle.load(open('hd.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    ft1 = request.form['age']
    ft2 = request.form['sex']
    ft3 = request.form['cp']
    ft4 = request.form['trestbps']
    ft5 = request.form['chol']
    ft6 = request.form['fbs']
    ft7 = request.form['restecg']
    ft8 = request.form['thalach']
    ft9 = request.form['exang']
    ft10 = request.form['oldpeak']
    ft11 = request.form['sl']
    ft12 = request.form['ca']
    ft13 = request.form['thal']
    arr = np.array([[ft1,ft2,ft3,ft4,ft5,ft6,ft7,ft8,ft9,ft10,ft11,ft12,ft13]])
    pred = model.predict(arr)


    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)















