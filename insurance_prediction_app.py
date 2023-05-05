from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

file = open("./random_forest_reg_model.pkl", 'rb')
model = pickle.load(file)

data = pd.read_csv('clean_insurance_data.csv')
data.head()

prediction = model.predict(pd.DataFrame([[36, 'female', 33.8, 0, 'no', 'northeast']],
                                        columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']))

print(prediction[0])


@app.route('/')
def index():
    return "Welcome !!!"


@app.route('/start')
def start():
    sex = sorted(data['sex'].unique())
    smoker = sorted(data['smoker'].unique())
    region = sorted(data['region'].unique())
    return render_template('insurance_prediction.html', sex=sex, smoker=smoker, region=region)


@app.route('/predict-premium', methods=['POST'])
def predictpremium():
    age = int(request.form.get('age'))
    sex = request.form.get('sex')
    bmi = float(request.form.get('bmi'))
    children = int(request.form.get('children'))
    smoker = request.form.get('smoker')
    region = request.form.get('region')

    prediction = model.predict(pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                                            columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']))

    return str(round(prediction[0], 2))


if __name__ == "__main__":
    app.run(debug=True)
