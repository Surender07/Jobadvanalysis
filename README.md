# Job Seeking Website with NLP and Machine Learning

## Overview
This repository hosts a job-seeking website, leveraging advanced machine learning and NLP techniques. The standout feature is the intelligent job category recommendation system, designed to enhance the experiences of both job seekers and employers.

## Key Features

### For Job Seekers
- Browse all open job adverts.
- View detailed job adverts from a summary view.
- Access jobs sorted by categories.

### For Employers
- Post new job listings with essential details.
- Utilize an NLP-driven category recommendation system for job listings, based on job title and description.
- Override the suggested categories for greater flexibility.

## NLP and Machine Learning Implementation
- **FastText Model**: Uses FastText for processing job descriptions, extracting meaningful features for classification.
- **Text Preprocessing**: Implements tokenization, stopwords removal, and lemmatization to prepare text data for analysis.
- **Logistic Regression for Category Prediction**: Employs a logistic regression model to predict job categories, providing accurate and relevant suggestions.
- **Interactive Job Posting**: Enables employers to post job listings, with the system offering category recommendations that can be accepted or overridden.

## Technologies Used
- Flask for the web framework.
- SQLAlchemy and Flask-Migrate for database management.
- Gensim's FastText for job description analysis.
- NLTK for natural language processing tasks like tokenization and lemmatization.
- Logistic Regression for predictive modeling.

## Installation
1. Clone the repository:
   ```git clone [repository-url]```
2. Install required dependencies:
   ```pip install -r requirements.txt```

## Demo:
To see the entire functionality of the website, please do watch the followin video:
`[https://youtu.be/jNXQ6nUCFi8](https://youtu.be/jNXQ6nUCFi8)`  


## Usage
1. Run the web application:
   ```python app.py```
2. Access the application at `[http://localhost:http://localhost:5000](http://localhost:5000)`.

## Contributing
Contributions, especially in the areas of NLP and machine learning, are welcome. Please follow the standard fork-and-pull request workflow.



