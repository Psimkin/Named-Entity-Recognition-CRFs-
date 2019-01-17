class SentenceGetter(object):

	def __init__(self, data):
		self.n_sent = 1
		self.data = data
		self.empty = False
		agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["word"].values.tolist(),
														   s["POS"].values.tolist(),
														   s["category"].values.tolist())]
		self.grouped = self.data.groupby("sentence_no").apply(agg_func)
		self.sentences = [s for s in self.grouped]

	def get_next(self):

		try:
			s = self.grouped["Sentence: {}".format(self.n_sent)]
			self.n_sent += 1
			return s
		except:
			return None


def word2features(sent, i):

	"""Standard CRF local feature extraction - using 'context window' features (i.e. words on either side of target word)"""

	word = sent[i][0]
	postag = sent[i][1]

	features = {
		'bias': 1.0,
		'word.lower()': word.lower(),
		'word[-3:]': word[-3:],
		'word[-2:]': word[-2:],
		'word.isupper()': word.isupper(),
		'word.istitle()': word.istitle(),
		'word.isdigit()': word.isdigit(),
		'postag': postag,
		'postag[:2]': postag[:2],
	}
	if i > 0:
		word1 = sent[i - 1][0]
		postag1 = sent[i - 1][1]
		features.update({
			'-1:word.lower()': word1.lower(),
			'-1:word.istitle()': word1.istitle(),
			'-1:word.isupper()': word1.isupper(),
			'-1:postag': postag1,
			'-1:postag[:2]': postag1[:2],
		})
	else:
		features['BOS'] = True

	if i < len(sent) - 1:
		word1 = sent[i + 1][0]
		postag1 = sent[i + 1][1]
		features.update({
			'+1:word.lower()': word1.lower(),
			'+1:word.istitle()': word1.istitle(),
			'+1:word.isupper()': word1.isupper(),
			'+1:postag': postag1,
			'+1:postag[:2]': postag1[:2],
		})
	else:
		features['EOS'] = True

	return features


def sent2features(sent):
	return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
	return [label for token, postag, label in sent]


def sent2tokens(sent):
	return [token for token, postag, label in sent]


def feature_extraction(df_train, df_test):
	"""Create X_train, X_test/y_train, y_test input"""

	# order/structure TRAINING sentences
	getter = SentenceGetter(df_train)
	sent = getter.get_next()
	sentences = getter.sentences

	# order/structure TESTING sentences
	getter_test = SentenceGetter(df_test)
	sent_test = getter_test.get_next()
	sentences_test = getter_test.sentences

	# transform train input into features/labels
	X_train = [sent2features(s) for s in sentences]
	y_train = [sent2labels(s) for s in sentences]

	# transform test input into features/labels
	X_test = [sent2features(s) for s in sentences_test]
	y_test = [sent2labels(s) for s in sentences_test]

	return X_train, X_test, y_train, y_test