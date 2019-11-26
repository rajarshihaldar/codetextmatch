# CodeTextMatch

Install the dependencies from environment.yml file using the command:
conda env create -f environment.yml

Download the following folders before running the models:

- data/ folder from https://drive.google.com/drive/folders/1H8TFDUzIFAezM1rVoB_1PQsCxNyqAQ2f?usp=sharing
- saved_models/ folder from https://drive.google.com/drive/folders/1EQwDw0JQzSXDEN5gaadN3LMxVyjDSYL8?usp=sharing

In src/ folder run train.py for training and eval.py in for evaluation

Default Config File: config.yml

- epochs: Number of epochs to train
- batch_size: Batch size to use
- model: Which Model to train/evaluate. Can be 'ct', 'cat' or 'mpctm'

To run train.py with default config file run:
- python train.py

To run train.py with custom config file run:
- python train.py -config path/to/config
- For example if your config file is ../config.yml use:
    - python train.py -config ../config.yml

Trained models are stored in saved_models/ folder and Evaluation results are stored in results/ folder