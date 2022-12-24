from flask import Flask,render_template,request,jsonify
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle
with open('../Models/Tockenizer.pickle',"rb") as f:
    Vectorizer=pickle.load(f)
app=Flask(__name__)
word_lemitizer=WordNetLemmatizer()
Regular_expression_definition_for_html_tags=re.compile('<.*?>')
Regular_expression_definition_for_digits=re.compile('\d+\s|\s\d+|\s\d+\s')
english_stop_words=stopwords.words('english')

def preprocessing_of_sentence(text):
    word_to_be_handled=[
    "not",
    "no",
    "very"
    ]
    text=Regular_expression_definition_for_html_tags.sub(r" ",text)
    text=Regular_expression_definition_for_digits.sub(r" ",text)
    punctuations = [".",",","!","?","'",'"',":",";","*","-","/","+","%","$","#","@","(",")","[","]","{","}"]
    for i in punctuations:
        text = text.replace(i," ")
    text=text.lower().split()
    text=[word for word in text if len(word)>1 and word not in english_stop_words or word in word_to_be_handled]
    text=[word_lemitizer.lemmatize(word) for word in text]
    # print(text)
    Vector=Vectorizer.transform([" ".join(text)])
    return Vector

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getresult',methods=['POST','GET'])
def getresult():
    print("getresult is requested")
    if request.method=='POST':
        print("Post Method")
        print(request.json['review'])
        if request.json['review']=="":
            return jsonify({"data":"Error occured"})
        else:
            input_=preprocessing_of_sentence(request.json['review'])
            print(input_)
            svmModel=joblib.load("../Models/Model.joblib")
            prediction=svmModel.predict(input_)[0]
            print(prediction)
            if prediction>=3:
                return jsonify({"data":"Good","rating":str(prediction)})
            elif prediction<=1:
                return jsonify({"data":"Bad","rating":str(prediction)})
            else:
                return jsonify({"data":"Neutral","rating":str(prediction)})
    else:
        return jsonify({"data":"Error occured"})


if __name__=='__main__':
    app.run(debug=True)