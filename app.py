#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
from flask import Flask, render_template, jsonify, request
import pandas as pd


# In[ ]:


app = Flask(__name__)
model = pickle.load(open('decisiontree.pkl','rb'))


# In[ ]:


@app.route('/')
def home():
    return render_template('solar.html')


# In[ ]:


yr = pd.Series([2017,2018,2019])
year = pd.DataFrame(yr, columns=['Year'])
year = pd.get_dummies(data=year,columns=['Year'],drop_first = True)
year = pd.concat([year,yr],1).set_index(0)
year


# In[ ]:


m = pd.Series([1,2,3,4,5,6,7,8,9,10,11,12])
month = pd.DataFrame(m,columns=['Month'])
month = pd.get_dummies(data=month,columns=['Month'],drop_first = True)
month = pd.concat([month,m],1).set_index(0)


# In[ ]:


d = pd.Series(['Day1-Day5','Day6-Day10','Day11-Day15','Day16-Day20','Day21-Day25','Day26-Day31'])
day = pd.DataFrame(d,columns=['Day'])
day = pd.get_dummies(data=day,columns=['Day'],drop_first = True)
day = pd.concat([day,d],1).set_index(0)
day


# In[ ]:


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == "POST":
        ct = float(request.form['ct'])
        dp = float(request.form['dp'])
        sza = float(request.form['sza'])
        sa = float(request.form['sa'])
        ws = float(request.form['ws'])
        ps = float(request.form['pw'])
        wd = float(request.form['wd'])
        rh = float(request.form['rh'])
        pressure = float(request.form['pressure'])
        atminutes = (float(request.form['hour']) * 60) + (float(request.form['min']))
        year1 = float(request.form['Year'])
        month1 = float(request.form['Month'])
        x = float(request.form['Day'])
        if x >=1 and x<=5:
            x = 'Day1-Day5'
        elif x>=6 and x<=10:
            x = 'Day6-Day10'
        elif x>=11 and x<=15:
            x= 'Day11-Day15'
        elif x>=16 and x<=20:
            x= 'Day16-Day20'
        elif x>=21 and x<=25:
            x= 'Day21-Day25'
        else: 
            x= 'Day26-Day31'
            
        for yr in year.index:
            if yr == year1:
                ydf = year.loc[yr]
            
        for mn in month.index:
            if mn == month1:
                mdf = month.loc[mn]
            
        for dy in day.index:
            if dy == x:
                ddf = day.loc[dy]
            
        feat = [np.array([ct,dp,sza,ws,ps,wd,rh,pressure,atminutes,*ydf.values,*mdf.values,*ddf.values])]
        pred = model.predict(feat)

        render_template("solar.html", prediction_text = 'Solar Radiation for the given time is {}'.format(pred))

        

                


# In[81]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




