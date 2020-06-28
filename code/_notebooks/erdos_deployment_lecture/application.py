
# imports related to the model we have built
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# imports related to flask and loading the model
from flask import Flask, request, jsonify
import pickle

json_header = {'content-type': 'application/json; charset=UTF-8'}
model = pickle.load(open('model.pickle', 'rb'))
app = Flask(__name__)    

@app.route('/', methods=['GET'])
def get_prediction():
    
    args = request.args

    # check that all args are present
    desired_args = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    missing_args = [a for a in desired_args if args.get(a) is None]
    
    if len(missing_args) > 0:
        error_msg = 'argument(s) missing: {}'.format(missing_args)
        return (jsonify(error_msg), 422, json_header)
                
    # check that all args are floats
    def arg_is_float(arg):
        is_float = False
        
        try:
            x = float(arg)
            is_float = True
        except ValueError:
            pass
        
        return is_float
    
    nonfloat_args = [a for a in desired_args if not arg_is_float(args.get(a))]
    
    if len(nonfloat_args) > 0:
        error_msg = 'argument(s) not float: {}'.format(nonfloat_args)
        return (jsonify(error_msg), 422, json_header)
    
    # make predictions
    X = [[float(args.get(a)) for a in desired_args]]
    prediction = str(model.predict(X)[0])
    
    return (jsonify({'prediction': prediction}), 200, json_header)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0')
