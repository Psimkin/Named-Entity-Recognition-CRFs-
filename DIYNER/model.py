from DIYNER import ner_processing
from DIYNER import crf_processing
import nltk
import pandas as pd
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics

class CRFNER(object):

	""" A class to get reviews for products on Amazon """

	def __init__(self, gazetteer, fraction=0.7):
		self.gazateer = gazetteer
		self.fraction = fraction

	def train(self, documents):
		self.data = ner_processing.NERFormatter(self.gazateer, documents)
		d_train, d_test = ner_processing.train_test_NER(self.data)

		self.X_train, self.X_test, self.y_train, self.y_test = crf_processing.feature_extraction(d_train, d_test)

		self.model = CRF(
			algorithm='lbfgs',
			c1=0.31,
			c2=0.02,
			max_iterations=100,
			all_possible_transitions=True)

		self.model.fit(self.X_train, self.y_train)

	def predict(self, sentence):
		"""Transforms a single sentence (for NER testing) into a CRF-suite format"""

		sentence_split = nltk.word_tokenize(sentence)
		n_words = [0] * len(sentence_split)

		df_pred = pd.DataFrame(
			{'word': sentence_split,
			 'sentence_no': n_words,
			 'category': n_words,
			 'POS': [x[-1] for x in nltk.pos_tag(sentence_split)],
			 })

		getter = crf_processing.SentenceGetter(df_pred)
		sent = getter.get_next()
		sentences = getter.sentences

		self.X = [crf_processing.sent2features(s) for s in sentences]
		return self.model.predict(self.X)

	def report(self):
		labels = list(self.model.classes_)

		y_pred = self.model.predict(self.X_test)
		print('F1 score {}'.format(metrics.flat_f1_score(self.y_test, y_pred,average='weighted', labels=labels)))

		sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))
		print(metrics.flat_classification_report(self.y_test, y_pred, labels=sorted_labels, digits=3))



