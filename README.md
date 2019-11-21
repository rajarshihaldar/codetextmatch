# CodeTextMatch

Install the dependencies from environment.yml file using the command:
conda env create -f environment.yml

Download the following folders before running the models:

- data/ folder from https://drive.google.com/drive/folders/1H8TFDUzIFAezM1rVoB_1PQsCxNyqAQ2f?usp=sharing
- saved_models/ folder from https://drive.google.com/drive/folders/1EQwDw0JQzSXDEN5gaadN3LMxVyjDSYL8?usp=sharing

Run the following files for training:

- ct_classifier.py: Uses Code and Text features as input (CT)
- cat_classifier.py: Uses Code, AST and Text features as input (CAT)
- mpctm_classifier.py: Uses Code, AST and Text features as input and uses multi-perspective architecture (MPCTM)

Run the following files for evaluation:

- eval_ct_classifier.py: Uses Code and Text features as input (CT)
- eval_cat_classifier.py: Uses Code, AST and Text features as input (CAT)
- eval_mpctm_classifier.py: Uses Code, AST and Text features as input and uses multi-perspective architecture (MPCTM)

Trained models are stored in saved_models/ folder and Evaluation results are stored in results/ folder