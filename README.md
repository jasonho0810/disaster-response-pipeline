# Disaster Response Pipeline Project
GitHub: https://github.com/jasonho0810/disaster-response-pipeline

## Project Motivation:
Following a disaster, there will often be millions of communications direct or via social media, just when disaster response organizations have the least capacities to filter and prioritize these messages. Often, only one in a thousand message requires the immediate attention of a disaster response professional.

Furthermore, different organizations typically take care of different aspects of the disaster. Our web app dashboard labels each communication message into labels that can be acted on by organizations. The labels are consistent over different disasters and can be used to investigate and visualize trends of these categories. The dashboard also shows trends on the message genres, message category labels, and the number of labels for the messages. 

Each message is labelled with 36 categories:
`'related' 'request' 'offer' 'aid_related' 'medical_help' 'medical_products' 'search_and_rescue' 'security' 'military' 'child_alone' 'water' 'food' 'shelter' 'clothing' 'money' 'missing_people' 'refugees' 'death' 'other_aid' 'infrastructure_related' 'transport' 'buildings' 'electricity' 'tools' 'hospitals' 'shops' 'aid_centers' 'other_infrastructure' 'weather_related' 'floods' 'storm' 'fire' 'earthquake' 'cold' 'other_weather' 'direct_report'`

Note: The dataset provided is imbalanced (certain labels like water have few examples). As such, the trained classifier model may produce naive classifications and misleading labellings. In this case, Accuracy of the model may be misleading and it may be best to focus more on Precision or Recall for these categories. If none of the messages are labelled with a certain category, Precision and F-score in the classification report will be ill-defined and set to 0 for that category.

## Instructions:
To run the web app, you can either host it locally or visit the website hosted on Heroku.
#### 1. Host it locally
1. Run the following commands in the project's root directory on command line to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://localhost:3001/

#### 2. Heroku
Go to Heroku website

## Installations:
```python
pip3 install -r requirements.txt
```
- pandas
- NumPy
- SQLalchemy
- NLTK - NLP data processing
- SciKit-Learn - Random Forest classifier
- Pickle5
- Plotly
- Flask

## File Description:
* `app/run.py` - Python script to host the web app locally
* `data/process_data.py` - Python script ETL pipeline to clean and store data in database
* `models/train_classifier.py` - Python script ML pipeline to train Random Forest classifier and save as a pickle file, and ouput model evaluation for each category

## Licensing, Authors, and Acknowledgements:
The base code is provided by Udacity as part of a project in their Data Scientist Nanodegree.
MIT License
