from flask import Flask,render_template, request , jsonify
from flask_cors import CORS,cross_origin
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/',methods = ['GET'])
@cross_origin()
def homePage():
	return render_template("index.html")

@app.route('/predict',methods = ['POST','GET'])
@cross_origin()

def index():
	if request.method == 'POST':
		try:
			is_occ_6 = (request.form['occ_6'])
			if (is_occ_6 == 'yes'):
				occ_6 = 1
			else:
				occ_6 = 0

			is_occ_husb_2 = (request.form['occ_husb_2'])
			if (is_occ_husb_2 == 'yes'):
				occ_husb_2 = 1
			else:
				occ_husb_2 = 0


			is_occ_husb_3 = (request.form['occ_husb_3'])
			if (is_occ_husb_3 == 'yes'):
				occ_husb_3 = 1
			else:
				occ_husb_3 = 0

			is_occ_husb_6 = (request.form['occ_husb_6'])
			if (is_occ_husb_6 == 'yes'):
				occ_husb_6 = 1
			else:
				occ_husb_6 = 0

			rate_marriage  = float(request.form['rate_marriage'])
			age  = float(request.form['age'])
			children  = float(request.form['children'])
			religious  = float(request.form['religious'])
			educ  = float(request.form['educ'])
			

			#load the scaler
			filename_1 = 'StandardScalar.sav'
			scaler_model = pickle.load(open(filename_1,'rb'))

			#transform the test data
			X_test_scaled = scaler_model.transform([[occ_6,occ_husb_2,occ_husb_3,occ_husb_6,rate_marriage,age,children,religious,educ]])
			print(X_test_scaled)

			#load the model
			filename_2  = 'modelForPrediction.sav'
			loaded_model = pickle.load(open(filename_2, 'rb')) #loading the model file from the storage

			#make predictions on the test set
			thr = 0.48480731163748536
			#prediction = loaded_model.predict(X_test_scaled)
			prediction = np.where(loaded_model.predict_proba(X_test_scaled)[:,1] >= thr, 1, 0) # .predict same as sklearn.predict ?
			if prediction[0] == 0:
				result = 'this women does not have an affair'
			else:
				result = 'this women has an affair'
			#print('prediction is', prediction[0])
			#Showing the prediction results in the UI
			return render_template('results.html',prediction=result)
		except Exception as e:
			print('The exception message is :',e)
			return'something is wrong'
	else:
		return render_template('index.html')

if __name__ == "__main__":
	app.run(debug = True)