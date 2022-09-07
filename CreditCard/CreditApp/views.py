from django.shortcuts import render
import pickle as pkl
import numpy as np

# Create your views here.
def index(request):
    return render(request,'index.html')

def load(fileName):
    file = open(fileName,'rb')
    data = pkl.load(file)
    file.close()
    return data

def predict(request):
    model = load('model.pkl')
    label = load('label.pkl')
    industry_onehot = load('industry_onehot.pkl')
    minmax = load('minmax.pkl')

    Gender = int(request.GET['Gender'])
    Age = request.GET['Age']
    Debt = request.GET['Debt']
    Married = int(request.GET['Married'])
    BankCustomer = int(request.GET['BankCustomer'])
    Industry = request.GET['Industry']
    YearsEmployed = request.GET['YearsEmployed']
    PriorDefault = int(request.GET['PriorDefault'])
    Employed = int(request.GET['Employed'])
    CreditScore = request.GET['CreditScore']
    DriversLicense = int(request.GET['DriversLicense'])
    Income = request.GET['Income']
    Approved = int(request.GET['Approved'])

    Industry1 = industry_onehot.transform([label.transform([Industry])]).toarray()
    test_data = np.array([[Gender,Age,Debt,Married,BankCustomer,YearsEmployed,PriorDefault,Employed,CreditScore,DriversLicense,Income]])
    test_X = np.c_[test_data,Industry1]
    test_X = minmax.transform(test_X)
    pred = model.predict(test_X)

    if pred[0]==1:
        msg = "FRAUD CASE !"
    else:
        msg = "NOT A FRAUD CASE"

    return render(request,'predict.html',{'prediction':msg})