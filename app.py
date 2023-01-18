import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

app = Flask(__name__)
model = pickle.load(open('catboost_regression.pickle', 'rb'))
cols=['make','model','year','engine_fuel_type','engine_hp','engine_cylinders','transmission_type','driven_wheels','number_of_doors','vehicle_size','vehicle_style','highway_mpg','city_mpg','popularity','crossover','diesel','exotic','factory_tuner','flex_fuel','hatchback','high-performance','hybrid','luxury','missing','performance']
@app.route('/')
def home():
    return render_template('template.html')

@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    feature_list = list(feature_list.values())
    feature_list = list(map(int, feature_list))
    final_features = np.array(feature_list).reshape(1, 25) 
    
    prediction = model.predict(final_features)
    output = prediction

    return render_template('index.html', prediction_text='Manufacturer Suggested Retail Price is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)