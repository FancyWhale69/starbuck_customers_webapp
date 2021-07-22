import json
from flask import Flask
from flask import render_template, request, jsonify
import joblib
import numpy as np
import os


app = Flask(__name__)

FILE_DIR = os.path.dirname(os.path.abspath('__file__'))
PARENT_DIR2 = os.path.join(FILE_DIR, 'models') 
# load model
model = joblib.load(PARENT_DIR2+'/model.pkl')



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # render web page 
    return render_template('index.html')

# age income bogo bogo disc disc disc disc info info total_tra F M 2014 2015 2016 2017 2018
    # age income  offer total_tra gender year
#person.append(float(query[0]))
    #person.append(float(query[1]))
    #person+=offer_onehotencode(query[2])
    #person.append(float(query[3]))
    #person+=gender_onehotencode(query[4])
    #person+=year_onehotencode(query[5])

# web page that handles user query and displays model results
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    # "Example: 20 2000 bogo_10_5 150 F 2015"
    temp= [x for x in request.form.values()]
    init_features = []
    init_features.append(float(temp[0]))
    init_features.append(float(temp[1]))
    init_features+=offer_onehotencode(temp[2])
    init_features.append(float(temp[3]))
    init_features+=gender_onehotencode(temp[4])
    init_features+=year_onehotencode(temp[5])
    

    final_features = [np.array(init_features)]

    prediction = model.predict(final_features) # making prediction


    return render_template('index.html', prediction_text='Positive Responce: {}'.format(prediction[0]==1))




def offer_onehotencode(offer_name):
    """
    return one-hot encoding for offers
    """
    offer=[0,0,0,0,0,0,0,0,0]
    if offer_name.lower() == 'bogo_10_5':
        return offer

    offers={'bogo_10_7':0,
            'bogo_5_5':1, 
            'bogo_5_7':2,
            'discount_10_10':3,
            'discount_10_7':4,
            'discount_20_10':5,
            'discount_7_7':6,
            'informational_0_3':7,
            'informational_0_4':8}

    offer[offers[offer_name.lower()]]+=1
    return offer

def gender_onehotencode(gender):
    """
    return one-hot encoding for gender
    """
    encode=[0,0]
    if gender.upper() == 'O':
        return encode

    genders={'F':0,
            'M':1}

    encode[genders[gender.upper()]]+=1
    return encode

def year_onehotencode(year):
    """
    return one-hot encoding for year
    """
    encode=[0,0,0,0,0]
    if year == '2013':
        return encode

    years={'2014':0,
            '2015':1, 
            '2016':2,
            '2017':3,
            '2018':4}

    encode[years[year]]+=1
    return encode

def main():
    app.run(port=5000, debug=True)


if __name__ == '__main__':
    main()