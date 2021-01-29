#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask,request,url_for,redirect,render_template,jsonify
import pandas as pd
import joblib
import numpy as np
import config


# In[ ]:


app=Flask(__name__)

model=joblib.load('LogisticRegression.pkl')

cols=config.Features

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[x for x in request.form.values()]
    final=np.array(int_features)
    data_unseen=pd.DataFrame([final],columns=cols)
    prediction=model.predict(data_unseen)
    return render_template('home.html',pred=" Species is {}.".format(prediction[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json(force=True)
    data_unseen=pd.DataFrame([data])
    prediction=model.predict(data_unseen)
    return jsonify(prediction)

if __name__=="__main__":
    app.run(debug=True)


# In[ ]:





# In[ ]:




