This is the code for the highest performing lexical simplification system featured on the paper:
"SIMPLEX-PB: A Lexical Simplification Database and Benchmark for Portuguese"

It contains three files:
- lib.py: A library with the classes and functions necessary to perform simplification.
- simplifier.py: A simple script that tests the simplifier.
- dataset_propor2018.txt: The test set used for the experiments featured in the paper.

To test the simplifier, run the following command:

python simplifier.py dataset_propor2018.txt <embeddings_model> <language_model> <how_many_to_generate>

The parameters are:
- <test_corpus>: A lexical simplification corpus in the victor format, which is the format of the "dataset_propor2018.txt" file. Each line contains a sentence, a target complex word, its index in the sentence, and a series of gold substitutions accompanied by their simplicity rank. To know more about the victor format, please visit the LEXenstein manual (https://github.com/ghpaetzold/LEXenstein).
- <embeddings_model>: A word embeddings model in the binary format produced by word2vec (https://radimrehurek.com/gensim/models/word2vec.html).
- <language_model>: A language model in the binary format produced by the KenLM toolkit (https://kheafield.com/code/kenlm).
- <how_many_to_generate>: The number of candidate substitutions that the model will generate for each target complex word.


This repository is result of the following paper:

```
Hartmann, Nathan S., Gustavo H. Paetzold, and Sandra M. Aluísio. "SIMPLEX-PB: A Lexical Simplification Database and Benchmark for Portuguese." International Conference on Computational Processing of the Portuguese Language. Springer, Cham, 2018.
```
