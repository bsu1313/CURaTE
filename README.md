# CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge

Unlearning method that works in real-time using retrieval of sentence embeddings.

## Usage

1. Create new conda environment and install requirements from requirements.txt
2. Run download_model.py to download model to local folder.
3. Run data_augmentation.py to create augmented dataset if it doesn't already exist.

### TOFU

#### Sentence Embeddings
1. Run train_sentemb.py to train the sentence embedding model.
2. Change path in "def get_available_cache_dir()" in evaluate_tofu_sentemb.py and run it.

### TruthfulQA: Refusal, Near Utility

#### Sentence Embeddings
1. Use sentence embedding model trained for TOFU.
2. Run truthfulQA/truthfulQA_evaluation_sentemb.py to evaluate the sentence embeddings on the TruthfulQA dataset.

### TruthfulQA: Far Utility

#### Sentence Embeddings
2. Use sentence embedding model trained for TOFU.
3. Run commonsense/evaluation_commonsenseQA_sentemb.py to evaluate the sentence embeddings on the CommonsenseQA dataset.

### Ablation
1. Run train_sentemb.py to train each baseline model with each ablation dataset
2. Run the files in the DB_files folder to generate mapping files for each ablation
3. Run the evaluation files with no_gen.py to get Precision, Recall, F1 scores for each ablation

### Multi-GPU (parallel)
#### Ubuntu
sudo apt-get update
sudo apt-get install -y libopenmpi-dev openmpi-bin

#### then in your conda/env:
pip install mpi4py

#### launch with:
DS_USE_MPI=0 deepspeed --num_gpus=4 truthfulQA_evaluation123_parallel.py
