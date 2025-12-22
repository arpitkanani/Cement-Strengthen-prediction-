from src.logger import logging
from src.exception import CustomException
from flask import Flask, render_template, request, redirect, url_for, session

import pandas as pd
import sys,os
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)
app=application

app.secret_key = "cement-secret-key"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        data = CustomData(
            cement=float(request.form.get('cement', 0)),
            blast_furance_slag=float(request.form.get('blast_furance_slag', 0)),
            fly_ash=float(request.form.get('fly_ash', 0)),
            water=float(request.form.get('water', 0)),
            superplasticizer=float(request.form.get('superplasticizer', 0)),
            coarse_aggregate=float(request.form.get('coarse_aggregate', 0)),
            fine_aggregate=float(request.form.get('fine_aggregate', 0)),
            age_in_day=int(request.form.get('age_in_day', 0))
        )

        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        
        prediction_pipeline=PredictPipeline()
        result=prediction_pipeline.predict(pred_df)
        session['result'] = round(result[0], 2)

            # 🔑 REDIRECT instead of render
        return redirect(url_for('predict_datapoint'))
    else:
        result = session.pop('result', None)
        return render_template('home.html', result=result)
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
    
 

