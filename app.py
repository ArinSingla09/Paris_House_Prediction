from flask import Flask, request, render_template,jsonify
import numpy as np
import pickle

# Fix the typo in the filename
model = pickle.load(open('HousePrdictionParis.pkl', 'rb'))

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])

    return render_template('home.html', prediction_text='House Price should be: {}'.format(output),input_text='Input data: {}'.format(int_features))
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
if __name__ == '__main__':
    app.run(debug=True)