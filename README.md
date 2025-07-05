# LTU Learning to Unlearn

Unlearning method using retrieval of sentence embeddings.


## Usage

1. Run data_augmentation.py to create augmented dataset if it doesn't already exist. (To train the classifier, we need examples of "Forget Information" being the same as "Query" so it learns to classify these as positive matches)
2. Run train_eval_roberta.py using augmented dataset to train the classifier.
3. Change path in "def get_available_cache_dir()" in evaluate_tofu_new.py and run it.


## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
