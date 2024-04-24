## Sentiment Analysis to Classify Restaurant Reviews - Research Paper

![Static Badge](https://img.shields.io/badge/💬-NLP-purple)
![Static Badge](https://img.shields.io/badge/📜-research_paper-blue)

Study with the aim of examining how user-generated metadata can enhance the accuracy and robustness of sentiment analysis models for classifying restaurant reviews.
Used [Joakim Arvidsson 10000 Restaurant Reviews dataset](https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews).
Model's trained/fine-tuned were 2 Random Forest models and a pre-trained BERT transformer model.

### Dependencies
[Torch](https://pytorch.org/) for ML tensor transformations.

[Transformers](https://huggingface.co/docs/transformers/en/index) for BERT classification and tokenization, and other necessary components like AdamW optimizer.

[Scikit learn](https://scikit-learn.org/) for train/test split, TfidfVectorizer, RandomForestClassifier and metric scores (accuracy_score, classification_report, confusion_matrix).

[NLTK](https://www.nltk.org/) for text tokenizing, stemming, lemmatization, removal of stop words, etc.

[Joblib](https://joblib.readthedocs.io/en/stable/) for model and tokenizer dumping and loading.

[Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for statistical data visualization.

[Time](https://docs.python.org/3/library/time.html), [datetime](https://docs.python.org/3/library/datetime.html#module-datetime) and [Babel](https://babel.pocoo.org/en/latest/dates.html)
for time related metrics.

### Download and run

```bash
# Clone repository
$ git clone https://github.com/smorenolasa1/NLP-group-assignment.git

# Upload to Google Colab or open with an IDE capable of Jupyter Notebooks

# If Google Colab isn't being used, we recommend removing the first cell:
!pip install transformers torch

# Execute each cell from top to bottom
```

### Contributing/License

Fork it, modify it, push it, eat it, summon a duck god with it. Whatever resonable day-to-day activity you prefer ( •ᴗ•)b
