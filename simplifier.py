from __future__ import print_function
import sys, os
from lib import *

#Reads data and parameters:
test_corpus = sys.argv[1]
embeddings_model_path = sys.argv[2]
language_model_path = sys.argv[3]
how_many = sys.argv[4]

#Creates generator and produces candidate substitutions:
eg = EmbeddingsGenerator(embeddings_model_path)
subs = eg.getSubstitutions(test_corpus, int(how_many))

#Creates a feature estimator:
fe = FeatureEstimator()
fe.addCollocationalFeature(language_model_path, 2, 2, 'Simplicity')
fe.addWordVectorSimilarityFeature(embeddings_model_path, 'Simplicity')
fe.addWordVectorContextSimilarityFeature(embeddings_model_path, 'Simplicity')

#Creates a ranker and ranks the candidates generated:
ar = AveragingRanker(fe)
text_data = toText(test_corpus, subs)
ranks = ar.getRankings(text_data)

#Evaluate generator:
ge = GeneratorEvaluator()
potential, precision, recall, fscore = ge.evaluateGenerator(test_corpus, subs)
print('Substitution Generation scores:')
print('\tPotential: ', potential)
print('\tPrecision: ', precision)
print('\tRecall: ', recall)
print('\tF-score: ', fscore)

#Evaluate full simplifier:
pe = PipelineEvaluator()
precision, accuracy, changed_proportion = pe.evaluatePipeline(test_corpus, ranks)
print('Full Pipeline scores:')
print('\tPrecision: ', precision)
print('\tAccuracy: ', accuracy)
print('\tChanged Proportion: ', changed_proportion)

