from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('iris_model.pkl', 'rb'))
@app.route('/', methods=['GET'])

def index():
    # Tampilkan form saja
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    sepal_l = float(request.form.get('sepal_l'))
    sepal_w = float(request.form.get('sepal_w'))
    petal_l = float(request.form.get('petal_l'))
    petal_w = float(request.form.get('petal_w'))
    # Masukkan ke model
    input_query = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    result = model.predict(input_query)[0]
    # Kembalikan ke template
    return render_template('form.html', result=str(result))

if __name__ == '__main__':
    app.run(debug=True)
    
