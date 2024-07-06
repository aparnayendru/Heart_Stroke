from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import sklearn as relief



app = Flask(__name__)


def preprocessing():
    global df, x_test, x_train, y_test, y_train
    df = pd.read_csv(r'C:\Users\ravinand\Downloads\TK69043\TK69043\DATA SET\heart.csv')

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=32)
    return x_train, x_test, y_train, y_test


def relieff():
    a = np.array(x_train)
    b = np.array(y_train)

    r = relief.Relief(n_features=13)
    tr = r.fit_transform(a, b)
    return relieff


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route('/prediction', methods=['GET','POST'])
def prediction():

    x_train, x_test, y_train, y_test = preprocessing()


    if request.method == "POST":

        model= int(request.form['algo'])
        if model == 0:
            dtc = DecisionTreeClassifier(random_state=4)
            dtc.fit(x_train, y_train)
            clf = dtc.predict(x_test)
            a=accuracy_score(clf, y_test)
            print(a)
            baglfya = BaggingClassifier(base_estimator=dtc, n_estimators=10)
            baglfya.fit(x_train, y_train)
            clfi = baglfya.predict(x_test)
            acrf = accuracy_score(y_test, clfi) * 100
            acrf = 'Accuracy of DecisionTree : ' + str(acrf)
            return render_template('prediction.html',msg=acrf ,acrf=acrf)
        elif model==1:
            x_train, x_test, y_train, y_test = preprocessing()
            rf = RandomForestClassifier(random_state=4, max_features=9, n_estimators=100)
            rf.fit(x_train, y_train)
            rfc = rf.predict(x_test)
            a=accuracy_score(y_test, rfc)
            baglfyb = BaggingClassifier(base_estimator=rf)
            baglfyb.fit(x_train, y_train)
            rfclf = baglfyb.predict(x_test)
            bcrf=accuracy_score(y_test,rfclf)*100
            bcrf = 'Accuracy of RandomForest: ' + str(bcrf)
            return render_template('prediction.html',msg=bcrf,bcrf=bcrf)
        elif model==2:
            x_train, x_test, y_train, y_test = preprocessing()
            knn = KNeighborsClassifier(n_neighbors=2)
            knn.fit(x_train, y_train)
            prediction = knn.predict(x_test)
            a=accuracy_score(prediction, y_test)
            baglfyc = BaggingClassifier(base_estimator=knn)
            baglfyc.fit(x_train, y_train)
            knnclf = baglfyc.predict(x_test)
            kcrf = accuracy_score(y_test, knnclf) * 100
            kcrf = 'Accuracy of KNN : ' + str(kcrf)
            return render_template('prediction.html',msg=kcrf,kcrf=kcrf)
        elif model==3:
            x_train, x_test, y_train, y_test = preprocessing()
            abc = AdaBoostClassifier(random_state=7)
            abc.fit(x_train, y_train)
            y_pred = abc.predict(x_test)
            abcrf = accuracy_score(y_test, y_pred) * 100
            abcrf = 'Accuracy of AdaBoost : ' + str(abcrf)
            return render_template('prediction.html',msg=abcrf,abcrf=abcrf)
        elif model==4:
            gbc = GradientBoostingClassifier(random_state=4)
            gbc.fit(x_train, y_train)
            pred = gbc.predict(x_test)
            gbcrf = accuracy_score(y_test, pred) * 100
            gbcrf = 'Accuracy of GradientBoosting:' + str(gbcrf)
            return render_template('prediction.html',msg=gbcrf,gbcrf=gbcrf)

        elif model==6:
            return render_template('prediction.html',msg="Please select a model")
    return render_template('prediction.html')


@app.route('/detection', methods=['GET','POST'])
def detection():
    if request.method == "POST":
        age=request.form['age']
        print(age)
        sex = request.form["sex"]
        print(sex)
        cp = request.form["cp"]
        print(cp)
        tresthps = request.form["tresthps"]
        print(tresthps)
        chol = request.form["cho"]
        print(chol)
        fbs = request.form["fbs"]
        print(fbs)
        restecg = request.form["restecg"]
        print(restecg)
        thalach = request.form["thalach"]
        print(thalach)
        exang = request.form["exang"]
        print(exang)
        oldpeak = request.form["oldpeak"]
        print(oldpeak)
        slope = request.form["slope"]
        print(slope)
        ca = request.form["ca"]
        print(ca)
        thal = request.form["thal"]
        print(thal)
        mna=[age, sex, cp, tresthps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        x_train, x_test, y_train, y_test = preprocessing()
        model= RandomForestClassifier(random_state=4, max_features=9, n_estimators=100)
        model.fit(x_train, y_train)
        output=model.predict([mna])
        print(output)
        if output==0:
            msg = 'The Patient have a <span style = color:red;>chance of getting Heart Stroke</span>'
        else:
            msg = 'The Patient have <span style = color:red;> No chance to get a Heart Stroke </span>'
        return render_template('detection.html',msg=msg)
    return render_template('detection.html')

if __name__ == '__main__':
    app.run(debug=True)
