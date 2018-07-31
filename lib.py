from __future__ import print_function
import gensim, nltk, codecs, kenlm

#Calculates the necessary features:
class FeatureEstimator:

	def __init__(self, norm=False):
		self.features = []
		self.identifiers = []
		self.norm = norm
		self.resources = {}
		self.temp_resources = {}
		
	def calculateFeatures(self, corpus, format='victor', input='file'):
		data = []
		if format.strip().lower()=='victor':
			if input=='file':
				data = [line.lower().strip().split('\t') for line in open(corpus)]
			elif input=='text':
				data = [line.lower().strip().split('\t') for line in corpus.split('\n')]
			else:
				print('Unrecognized format: must be file or text.')
		elif format.strip().lower()=='cwictor':
			if input=='file':
				f = open(corpus)
				for line in f:
					line_data = line.strip().split('\t')
					data.append([line_data[0].strip(), line_data[1].strip(), line_data[2].strip(), '0:'+line_data[1].strip()])
			elif input=='text':
				for line in corpus.split('\n'):
					line_data = line.strip().split('\t')
					data.append([line_data[0].strip(), line_data[1].strip(), line_data[2].strip(), '0:'+line_data[1].strip()])
			else:
				print('Unrecognized format: must be file or text.')
		else:
			print('Unknown input format during feature estimation!')
			return []


		values = []
		for k, feature in enumerate(self.features):
			values.append(feature[0].__call__(data, feature[1]))
		
		result = []
		index = 0
		for line in data:
			for i in range(3, len(line)):
				vector = self.generateVector(values, index)
				result.append(vector)
				index += 1
		if self.norm:
			result = normalize(result, axis=0)
			
		self.temp_resources = {}

		return result
		
	def calculateInstanceFeatures(self, sent, target, head, candidate):
		data = [[sent, target, head, '0:'+candidate]]
		
		values = []
		for feature in self.features:
			values.append(feature[0].__call__(data, feature[1]))
		vector = self.generateVector(values, 0)
		return vector
		
	def generateVector(self, feature_vector, index):
		result = []
		for feature in feature_vector:
			if not isinstance(feature[index], list):
				result.append(feature[index])
			else:
				result.extend(feature[index])
		return result
		
	def getNgram(self, cand, tokens, head, configl, configr):
		if configl==0 and configr==0:
			return cand, False, False
		else:
			result = ''
			bosv = False
			if max(0, head-configl)==0:
				bosv = True
			eosv = False
			if min(len(tokens), head+configr+1)==len(tokens):
				eosv = True
			for i in range(max(0, head-configl), head):
				result += tokens[i] + ' '
			result += cand + ' '
			for i in range(head+1, min(len(tokens), head+configr+1)):
				result += tokens[i] + ' '
			return result.strip(), bosv, eosv
	
	def addCollocationalFeature(self, language_model, leftw, rightw, orientation):
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if language_model not in self.resources:
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			self.features.append((self.collocationalFeature, [language_model, leftw, rightw]))
			for i in range(0, leftw+1):
				for j in range(0, rightw+1):
					self.identifiers.append(('Collocational Feature ['+str(i)+', '+str(j)+'] (LM: '+language_model+')', orientation))

	def collocationalFeature(self, data, args):
		lm = args[0]
		spanl = args[1]
		spanr = args[2]
		result = []
		model = self.resources[lm]
		for line in data:
			sent = line[0].strip().split(' ')
			target = line[1]
			head = int(line[2])
			spanlv = range(0, spanl+1)
			spanrv = range(0, spanr+1)
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				values = []
				for span1 in spanlv:
					for span2 in spanrv:
						ngram, bosv, eosv = self.getNgram(word, sent, head, span1, span2)
						aux = model.score(ngram, bos=bosv, eos=eosv)
						values.append(aux)
				result.append(values)
		return result

	def addWordVectorSimilarityFeature(self, model, orientation):
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if model not in self.resources:
				m = gensim.models.KeyedVectors.load_word2vec_format(model, binary=True)
				self.resources[model] = m
			self.features.append((self.wordVectorSimilarityFeature, [model]))
			self.identifiers.append(('Word Vector Similarity (Model: '+model+')', orientation))
		
	def wordVectorSimilarityFeature(self, data, args):
		model = self.resources[args[0]]
		result = []
		for line in data:
			target = line[1].strip().lower()
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				similarity = 0.0
				cand_size = 0
				for word in words.split(' '):
					cand_size += 1
					try:
						similarity += model.similarity(target, word)
					except KeyError:
						try:
							similarity += model.similarity(target, word.lower())
						except KeyError:
							pass
				similarity /= cand_size
				result.append(similarity)
		return result
	
	def addWordVectorContextSimilarityFeature(self, model, orientation):
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if model not in self.resources:
				m = gensim.models.KeyedVectors.load_word2vec_format(model, binary=True)
				self.resources[model] = m
			self.features.append((self.wordVectorContextSimilarityFeature, [model]))
			self.identifiers.append(('Word Vector Context Similarity (Model: '+model+')', orientation))
	
	def wordVectorContextSimilarityFeature(self, data, args):
		model = self.resources[args[0]]
		result = []
		
		for i in range(0, len(data)):
			line = data[i]
			tokens = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			
			content_words = set([])
			bleft = max(0, head-3)
			eright = min(len(tokens), head+4)
			content_words.update(tokens[bleft:head])
			content_words.update(tokens[head+1:eright])

			for subst in line[3:len(line)]:
				word = subst.strip().split(':')[1].strip()
				similarity = 0.0
				for content_word in content_words:
					try:
						similarity += model.similarity(content_word, word)
					except KeyError:
						try:
							similarity += model.similarity(content_word, word.lower())
						except KeyError:
							pass
				similarity /= float(len(content_words))
				result.append(similarity)
		return result

#Generates candidates using embeddings:
class EmbeddingsGenerator:

	def __init__(self, w2vmodel):
		self.stemmer = nltk.stem.RSLPStemmer()
		self.model = gensim.models.KeyedVectors.load_word2vec_format(w2vmodel, unicode_errors='ignore', binary=True)

	def getSubstitutions(self, victor_corpus, amount):
		substitutions = self.getInitialSet(victor_corpus, amount)
		return substitutions

	def getInitialSet(self, victor_corpus, amount):
		lexf = codecs.open(victor_corpus, encoding='utf8')
		data = []
		for line in lexf:
			d = line.strip().split('\t')
			data.append(d)
		lexf.close()
		
		trgs = []
		for i in range(0, len(data)):
			d = data[i]
			target = d[1].strip().lower()
			head = int(d[2].strip())
			trgs.append(target)
	
		subs = []
		cands = set([])
		for i in range(0, len(data)):
			d = data[i]

			t = trgs[i]

			word = t

			most_sim = []
			try:
				most_sim = self.model.most_similar(positive=[word], topn=50)
			except KeyError:
				most_sim = []

			subs.append([word[0] for word in most_sim])
			
		subs_filtered = self.filterSubs(data, subs, trgs)
		
		final_cands = {}
		for i in range(0, len(data)):
			target = data[i][1]
			cands = subs_filtered[i][0:min(amount, len(subs_filtered[i]))]
			cands = [word.split('|||')[0].strip() for word in cands]
			if target not in final_cands:
				final_cands[target] = set([])
			final_cands[target].update(set(cands))
		
		return final_cands
		
	def lemmatizeWords(self, words):
		result = []
		for word in words:
			result.append(self.lemmatizer.lemmatize(word))
		return result
		
	def stemWords(self, words):
		result = []
		for word in words:
			result.append(self.stemmer.stem(word))
		return result
	
	def filterSubs(self, data, subs, trgs):
		result = []
		for i in range(0, len(data)):
			d = data[i]

			t = trgs[i]
			tstem = self.stemmer.stem(t)
			word = t

			most_sim = subs[i]
			most_simf = []

			for cand in most_sim:
				cword = cand
				cstem = self.stemmer.stem(cword)

				if cword not in word and word not in cword and cstem not in word and tstem not in cword:
					most_simf.append(cand)

			result.append(most_simf)
		return result

#Ranks candidates using rank averaging:
class AveragingRanker:

	def __init__(self, fe):
		self.fe = fe
		self.feature_values = None
		
	def getRankings(self, victor_corpus):
		self.feature_values = self.fe.calculateFeatures(victor_corpus, input='text')
		
		result = []
		
		index = 0
		for line in victor_corpus.split('\n'):
			data = line.strip().split('\t')
			substitutions = data[3:len(data)]
			
			instance_features = []
			for substitution in substitutions:
				instance_features.append(self.feature_values[index])
				index += 1
			
			rankings = {}
			for i in range(0, len(self.fe.identifiers)):
				scores = {}
				for j in range(0, len(substitutions)):
					substitution = substitutions[j]
					word = substitution.strip().split(':')[1].strip()
					scores[word] = instance_features[j][i]
				
				rev = False
				if self.fe.identifiers[i][1]=='Simplicity':
					rev = True
				
				words = scores.keys()
				sorted_substitutions = sorted(words, key=scores.__getitem__, reverse=rev)
				
				for j in range(0, len(sorted_substitutions)):
					word = sorted_substitutions[j]
					if word in rankings:
						rankings[word] += j
					else:
						rankings[word] = j
		
			final_rankings = sorted(rankings.keys(), key=rankings.__getitem__)
		
			result.append(final_rankings)
		return result
		
	def size(self):
		return len(self.fe.identifiers)

#Transforms a substitution dictionary into a victor dataset:		
def toText(victor_corpus, subs):
	f = codecs.open(victor_corpus, encoding='utf8')
	result = ''
	for line in f:
		data = line.strip().split('\t')
		target = data[1].strip()
		newline = '\t'.join(data[:3])
		if target in subs:
			for cand in subs[target]:
				newline += '\t0:'+cand
		if len(newline.strip())>0:
			result += newline + '\n'
	f.close()
	return result.strip()


#Evaluates a Substitution Generation approach:
class GeneratorEvaluator:

	def evaluateGenerator(self, victor_corpus, substitutions):
		potentialc = 0
		potentialt = 0
		precisionc = 0
		precisiont = 0
		recallt = 0
		
		f = open(victor_corpus)
		for line in f:
			data = line.strip().split('\t')
			target = data[1].strip()
			items = data[3:len(data)]
			candidates = set([item.strip().split(':')[1].strip() for item in items])
			if target in substitutions:
				overlap = candidates.intersection(set(substitutions[target]))
				precisionc += len(overlap)
				if len(overlap)>0:
					potentialc += 1
				precisiont += len(substitutions[target])
			potentialt += 1
			recallt += len(candidates)
		f.close()
		
		potential = float(potentialc)/float(potentialt)
		precision = float(precisionc)/float(precisiont)
		recall = float(precisionc)/float(recallt)
		fmean = 0.0
		if precision==0.0 and recall==0.0:
			fmean = 0.0
		else:
			fmean = 2*(precision*recall)/(precision+recall)
			
		return potential, precision, recall, fmean

#Evaluates a full simplifier:
class PipelineEvaluator:

	def evaluatePipeline(self, victor_corpus, rankings):
		total = 0
		totalc = 0
		accurate = 0
		precise = 0
		
		f = codecs.open(victor_corpus, encoding='utf8')
		for i in range(0, len(rankings)):
			data = f.readline().strip().split('\t')
			target = data[1].strip()
			data = data[3:len(data)]
			gold_subs = set([item.strip().split(':')[1].strip() for item in data])
			
			if len(rankings[i])>0:
				first = rankings[i][0]
			else:
				first = '|||NULL|||'
			
			total += 1
			if first!=target:
				totalc += 1
				if first in gold_subs:
					accurate += 1
					precise += 1
			else:
				precise += 1
		
		return float(precise)/float(total), float(accurate)/float(total), float(totalc)/float(total)
