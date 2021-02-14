from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
model = pickle.load(open('11.pkl', 'rb'))

# # df.Species = le.fit_transform(df['Species'])
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()

#
# from flask import Flask, render_template, request

app = Flask(__name__)


# @app.route('/')
@app.route('/')
def man():
    return render_template('home.html')




@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['h']
    data13 = request.form['i']
    data14 = request.form['j']
    data15 = request.form['k']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15]])
    pred = model.predict(arr)
    return render_template('after.html', data2=pred)


if __name__ == "__main__":
    app.run(debug=True)















