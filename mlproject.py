 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import re
import tkinter as tk
 
true = pd.read_csv('True.csv', encoding='unicode_escape')
fake = pd.read_csv('Fake.csv')
true['label']=1
fake['label'] = 0

news = pd.concat([fake,true],axis=0)
news.isnull().sum()
 
news = news.drop(['title','subject','date'],axis=1)
news1 = news.sample(frac = 1)   #shuffling
news.reset_index(inplace=True)
news.drop(['index'],axis=1,inplace=True)

def wordopt(text):
    #convert into lowercase
    text = text.lower()
    
    #remove urls 
    text = re.sub(r'https?://|S+|www\.\S+','',text)
    
    #remove html tags
    text = re.sub(r'<.*?>','',text)
    
    #remove punctuation
    text = re.sub(r'[^\w\s]','',text)
    
    #remove digits
    text = re.sub(r'\d','',text)
    
    #remove newline characters
    text = re.sub(r'\n',' ',text)
    return text

 
news['text'] = news['text'].apply(wordopt)
x = news['text']
y = news['label']

 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#vectorization feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
print("vectorization start..")
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
print("vectorization complete ...")

#logistic regression
from sklearn.linear_model import LogisticRegression
print("Logistic regresssion Training start....")
lr = LogisticRegression()
lr.fit(xv_train,y_train)
pred_lr = lr.predict(xv_test)
lr.score(xv_test,y_test)
from sklearn.metrics import classification_report
print("Logistic Regression classification report : ")
print(classification_report(y_test,pred_lr))


# #Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
print("Random Forest Classifier Training start....")
rfc = RandomForestClassifier()
rfc.fit(xv_train,y_train)
predict_rfc = rfc.predict(xv_test)
rfc.score(xv_test,y_test)
print("Classification Report of Random Forest Classifier : ")
print(classification_report(y_test,predict_rfc))


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
print("Gradient Boosting Classifier Training start....")
gbc = GradientBoostingClassifier()
gbc.fit(xv_train,y_train)
pred_gbc = gbc.predict(xv_test)
print("Classification Report of Gradient Boosting Classifier : ")
print(classification_report(y_test,pred_gbc))


def output_label(n):
    if n==0:
        return "it is a fake news"
    elif n == 1:
        return "it is a genuine news"

 
def manual_testing(news):
    testing_news = {"text":[news]} 
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_lr = lr.predict(new_xv_test)
    res = ""
    if(pred_lr[0]==0):
         res = res + " LR Prediction : it is a fake news\n"
    elif pred_lr[0] == 1:
        res=res+" LR Prediction : it is a genuine news\n"
    pred_gbc = gbc.predict(new_xv_test)
    if(pred_gbc[0]==0):
         res = res + " GBC Prediction : it is a fake news\n"
    elif pred_gbc[0] == 1:
        res=res+" GBC Prediction : it is a genuine news\n"
    pred_rfc = rfc.predict(new_xv_test)
    if(pred_rfc[0]==0):
         res = res + " RFC Prediction : it is a fake news"
    elif pred_rfc[0] == 1:
        res=res+" RFC Prediction : it is a genuine news"

    return res


import tkinter as tk
from tkinter import ttk

def process_input():
    user_input = input_entry.get()
    result = manual_testing(user_input)
    output_label.config(text=f"Output:\n {result}")

def on_close():
    window.destroy()
window = tk.Tk()
window.title("Fake News Detection Application")
style = ttk.Style()
style.theme_use("clam")
frame = ttk.Frame(window)
frame.pack(pady=20, padx=20)
input_label = ttk.Label(frame, text="Enter article:", font=("Arial", 12))
input_label.pack(side=tk.LEFT, padx=5)
input_entry = ttk.Entry(frame, font=("Arial", 12))
input_entry.pack(side=tk.LEFT, padx=5)
process_button = ttk.Button(frame, text="Process", command=process_input)
process_button.pack(side=tk.LEFT, padx=5)
output_label = ttk.Label(window, text="Output: ", font=("Arial", 12))
output_label.pack(pady=20, padx=20)
window.protocol("WM_DELETE_WINDOW", on_close)
window.mainloop()
