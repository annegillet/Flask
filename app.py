from flask import Flask, render_template, request, redirect, url_for, send_file
import csv
import pandas as pd
import matplotlib.image as mpimg
from keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import mysql.connector as MS
from werkzeug import secure_filename
from skimage import io, color
import os
import pickle

#============================== Q.1 à 6 ===============================

connection = MS.connect(user='root', password='rootroot', host='127.0.0.1', buffered=True)
cursor = connection.cursor()

utiliser_bd = "USE Flask"
cursor.execute(utiliser_bd)

app = Flask(__name__)

app.config.update(DEBUG=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/test-formulaire')
def contact_page():
    return render_template("home.html")

@app.route('/test-formulaire', methods=['POST'])
def user():
    nom = request.form['nom']
    prenom = request.form['prenom']
    sexe = request.form['sexe']
    pseudo = request.form['pseudo']

    if sexe == 'Féminin':
        texte = f"Bonjour Mme {prenom}, {nom}, votre nom d'utilisateur est {pseudo}"
    else:
        texte = f"Bonjour Mr {prenom}, {nom}, votre nom d'utilisateur est {pseudo}"

    pseudo_existant = "SELECT * FROM users WHERE pseudo = '%s' "

    cursor.execute(pseudo_existant % pseudo)
    resultat_pseudo_existant = cursor.fetchall()
    print(resultat_pseudo_existant)

    if len(resultat_pseudo_existant) > 0:
        error = 'Ce pseudo est déjà utilisé, veuillez utiliser un autre pseudo'
        return render_template("bienvenue.html", error = error)

    else:
        enregister_user = "INSERT INTO users (Prenom, Nom, Sexe, Pseudo) VALUES (%s,%s,%s,%s)"
        cursor.execute(enregister_user, (prenom, nom, sexe, pseudo))
        connection.commit()

    return render_template("bienvenue.html", message= texte)

@app.route('/utilisateurs-inscrits')
def inscrits():
    infos_user = "SELECT nom FROM users "
    cursor.execute(infos_user)
    resultat_infos_user = cursor.fetchall()
    return render_template("/inscrits.html", infos_user=resultat_infos_user)

#============================== Q.7 ===============================

@app.route('/stat')
def stat_page() :
    return render_template('stat-df.html')

@app.route('/stat', methods=['POST'])    
def load():
    file = request.files['file']
    file.save('data/file.csv')

    df = pd.read_csv('data/file.csv') 
    desc = df.describe()
    
    return render_template('stat-df.html', tables=[desc.to_html(classes="data")])

#============================== Q.8 ===============================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Définition et Formatage du train et du test
X_train = X_train.reshape(X_train.shape[0],-1)
X_test = X_test.reshape(X_test.shape[0],-1)

@app.route('/pred')
def pred_page() :
    return render_template('pred.html')

@app.route('/pred', methods=['POST'])
def mnist():
    #Récupération de l'image
    img = request.files['image']
    img_3d = io.imread(img)
    img_2d = color.rgb2gray(img_3d)
    img_2d = img_2d.reshape(1, -1)

    #Random Forest
    #rf = pickle.load(open('mnist_rf.sav', 'rb'))
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    #Logistic Regression
    #reglog = pickle.load(open('mnist_reglog.sav', 'rb'))
    #reglog = LogisticRegression()
    #reglog.fit(X_train, y_train)

    #XG Boost - Trop long et pas si pertinent
    #xg = XGBClassifier()
    #xg.fit(X_train, y_train)

    #Prédiction des résultats de l'image
    y_pred_rf = rf.predict(img_2d)
    score_rf = round(rf.score(X_test, y_test)*100,2)

    #y_pred_reglog = reglog.predict(img_2d)
    #score_reglog = reglog.score(X_test, y_test)

    #y_pred_xg = xg.predict(img_2d)
    #y_score_xg = [y_pred_xg[0]]
    #for n in range(1, len(y_pred_xg)):
    #    if y_pred_xg[n] == y_pred_xg[0]:
    #        y_score_xg.append(y_pred_xg[n])
    #score_xg = round((len(y_score_xg)/len(y_pred_xg))*100,2)

    #score = classification_report(y_test, y_pred)

    return render_template('pred.html', score_rf= score_rf, y_pred_rf=y_pred_rf)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)