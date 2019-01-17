import nltk
import pandas as pd
from DIYNER.cleaning import cleantext
import itertools
import random

def EntityTagger(gazateer, documents):

	""" Create matches between gazetteer and documents """

	# Split input text into individual sentences and remove single words/empty sentences
	sentences = [nltk.sent_tokenize(txt) for txt in documents]
	sentences = list(itertools.chain(*sentences))

	# Create dictionary and append matches in sentences to entities
	tagged = {}

	for sentence in sentences:
		tagged[sentence] = []
		for value, ent in gazateer.items():
			if value.lower() in sentence.lower():
				tagged[sentence].append(value)

	# Delete dictionary entries that are empty
	for key in tagged.copy().keys():
		if not tagged[key]:
			del tagged[key]
		else:
			cleankey = cleantext(key)
			tagged[cleankey] = tagged.pop(key)

	return tagged

def NERFormatter(gazetteer, documents):

	""" Structure training data """

	tagged = EntityTagger(gazetteer, documents)
	sentence_no = 0
	results = {}

	for item, ners in tagged.items():
		sentence_no += 1
		for ner in ners:
			sentence_split = nltk.word_tokenize(item)
			pos_tags = [x[-1] for x in nltk.pos_tag(sentence_split)]
			# get index of string/partial string match
			index_of_ngram = [i for i, word in enumerate(sentence_split) if any(x in word for x in nltk.word_tokenize(ner))]
			# create frame of sequence
			ner_frame = ['0'] * len(sentence_split)
			# input entity where index = match
			for idx in index_of_ngram:
				ner_frame[idx] = ner
				results[item] = {'word':sentence_split,'entity': ner_frame,'sentence_no': [sentence_no] * len(sentence_split),'POS':pos_tags}

	data = pd.DataFrame(results).T
	data = (data.set_index(data.index).apply(lambda x: x.apply(pd.Series).stack()).reset_index().drop('level_1', 1))
	data['category'] = data['entity'].map({v: k for v, k in gazetteer.items()})
	data.fillna('0', inplace=True)
	data.drop('level_0', inplace=True, axis=1)

	return data

def train_test_NER(data,fraction=0.7):
	unique_sentences_numbers = int(data['sentence_no'].nunique())
	split_fraction = int(unique_sentences_numbers * fraction)
	random_sample = random.sample(range(0, unique_sentences_numbers), split_fraction)
	d_train = data.loc[data['sentence_no'].isin(random_sample)]
	d_test = data.loc[~data['sentence_no'].isin(random_sample)]
	return d_train, d_test