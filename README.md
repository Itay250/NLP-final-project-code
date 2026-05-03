This README.md provides an overview of the core components in this repository, which explores the "Trust Gap" between a model's internal certainty and its linguistic decisiveness.
This project investigates how instruction-tuning and persona-driven prompting influence model calibration, specifically examining the divergence between mathematical log-probabilities and authoritative phrasing.

File Explanations
Data and Inference Results
test.csv: Contains the FalseQA dataset used for evaluation. This dataset consists of adversarial questions with false premises designed to trap models into generating hallucinations.

pythia_results.json, qwen_results.json: Stores the raw JSON results from our inference runs. This includes the model's generated text, the prompt used (Neutral vs. Expert), and the associated accuracy scores.

Core Logic and Scripts
inference.py: The main script for deploying Qwen-2.5-0.5B-Instruct and Pythia-410M. It handles the local extraction of raw logits and normalized log-probabilities using the Hugging Face library.

calculations.py: Implements the logic for our evaluation framework.

Visual Findings
Figure_pythia.png, Figure_qwen.png: final visual results from the test, each file contains 4 graphs.
