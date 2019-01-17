from DIYNER.model import CRFNER
import pandas as pd

# Create Gazateer
gazetteer = pd.read_csv('gazetteer/gazateer.csv')
gazetteer = dict(zip(gazetteer['entities'].tolist(), gazetteer['categories'].tolist()))

# Get documents
documents = [str(x) for x in pd.read_csv('data/techCorpus.csv')['text'].tolist()][0:5]

# Training Model
ner_crf = CRFNER(gazetteer)
ner_crf.train(documents)

# Predictions
ner_crf.predict('Google are opening new stores in Vancouver')