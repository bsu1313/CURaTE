# LTU Learning to Unlearn

Unlearning method using retrieval of sentence embeddings.

## Usage

### TOFU

#### Classifier
1. Run data_augmentation.py to create augmented dataset if it doesn't already exist. (To train the classifier, we need examples of "Forget Information" being the same as "Query" so it learns to classify these as positive matches)
2. Run train_roberta.py using augmented dataset to train the classifier.
3. Change path in "def get_available_cache_dir()" in evaluate_tofu_classifier.py and run it.

#### Sentence Embeddings
1. Run train_sentemb.py to train the sentence embedding model.
2. Change path in "def get_available_cache_dir()" in evaluate_tofu_sentemb.py and run it.


### TruthfulQA: Refusal, Near Utility

#### Classifier
1. Use classifier trained for TOFU.
2. Run truthfulQA/truthfulQA_evaluation123.py to evaluate the classifier on the TruthfulQA dataset.

#### Sentence Embeddings
1. Use sentence embedding model trained for TOFU.
2. Run truthfulQA/truthfulQA_evaluation_sentemb.py to evaluate the sentence embeddings on the TruthfulQA dataset.


### TruthfulQA: Far Utility
#### Classifier
1. Use classifier trained for TOFU.
#### Sentence Embeddings
2. Use sentence embedding model trained for TOFU.