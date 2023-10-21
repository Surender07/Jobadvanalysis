from unicodedata import category
from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import datetime
import numpy as np
from gensim.models.fasttext import FastText
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string


def preprocess_text(text, method='lemmatize'):
    """
    Perform basic NLP preprocessing on a given text.
    
    Args:
    - text (str): Input text string to be preprocessed.
    - method (str): Either 'lemmatize' or 'stem'. Default is 'lemmatize'.
    
    Returns:
    - list: List of preprocessed tokens.
    """
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation and numbers
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming or Lemmatization
    if method == 'stem':
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    elif method == 'lemmatize':
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    else:
        raise ValueError("Method argument should be either 'stem' or 'lemmatize'")
    
    return tokens


app = Flask(__name__, static_folder='styles', static_url_path='/styles')

# Adding configuration for using a sqlite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Creating an SQLAlchemy instance
db = SQLAlchemy(app)

# Settings for migrations
migrate = Migrate(app, db)


def docvecs(embeddings, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        docvec = np.vstack([embeddings[term] for term in valid_keys])
        docvec = np.sum(docvec, axis=0)
        vecs[i,:] = docvec
    return vecs

def modify_preds(y_pred):
    pred = None
    if y_pred == "Accounting_Finance":
        pred = "Accounting Finance"
    if y_pred == "Healthcare_Nursing":
        pred = "Healthcare Nursing"
    if y_pred == "Sales":
        pred = "Sales"
    if y_pred == "Engineering":
        pred = "Engineering"
    return pred



# Models
class advertisment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(20), unique=False, nullable=True)
    description = db.Column(db.String(1000), unique=False, nullable=False)
    salary = db.Column(db.String(50), nullable=False)
    job_category = db.Column(db.String(20), unique=False, nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.datetime.utcnow)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        category = request.form.get('category')
        ads = advertisment.query.filter(advertisment.job_category == category).all()
        return render_template('index.html', title="jobs", ads=ads)
    return render_template('index.html')

@app.route("/manager")
def manager():
    return render_template('manager.html')

@app.route('/job/<int:job_id>')
def job_details(job_id):
    # Fetch job details from your database based on job_id
    job = advertisment.query.get(job_id)
    if job:
        return render_template('job_details.html', ad=job)
    else:
        return "Job not found", 404


@app.route("/post_data", methods=['GET', 'POST'])
def post_data():
    if request.method == "POST":
        job_title_post = request.form.get('job_title', '')
        job_desc = request.form.get('job_desc', '')
        radio_post = request.form.get('suggested_choice', '')
        salary_post = request.form.get('salary', '')
        
        # Load logistic regression pre-trained model
        descFT = FastText.load("models/desc_FT.model")
        descFT_wv= descFT.wv
        tokenized_data = preprocess_text(job_desc)
        bbcFT_dvs = docvecs(descFT_wv, [tokenized_data])

        # Load the LR model
        pkl_filename = "models/descFT_LR.pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)

        # Predict the label of tokenized_data
        y_pred = model.predict(bbcFT_dvs)
        y_pred = y_pred[0]

        print(y_pred)

        # Convert sentence to list to be used by the model
        job_desc = [job_desc]
    
        
        if radio_post == "true":
            ad_object = advertisment(title=job_title_post, description=job_desc[0], salary=salary_post, job_category=(y_pred))
            db.session.add(ad_object)
            db.session.commit()
            feedback = "success"
            return jsonify(feedback=feedback)
        elif radio_post == "false":
            custom_category = request.form.get('custom_category', '')
            ad_object = advertisment(title=job_title_post, description=job_desc[0], salary=salary_post, job_category=custom_category)
            db.session.add(ad_object)
            db.session.commit()
            feedback = "success"
            return jsonify(feedback=feedback)
        else:
            return jsonify(category=y_pred)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
