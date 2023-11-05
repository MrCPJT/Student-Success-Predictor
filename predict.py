# Pickle to read model/dv file
import pickle

# Flask to serve model 
from flask import Flask
from flask import request
from flask import jsonify

# Reading the file

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

# Decorator (POST method - retrieves JSON information from user)
@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict(X)
    
    result = {
        'outcome': float(y_pred)
    }
    
    return jsonify(result)

# Running the webservice
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=2222)