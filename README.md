# IIT Patna Student Life Question Answering System

## Overview

This project implements a Question Answering (QA) system specifically designed to answer queries about student life at the Indian Institute of Technology Patna (IIT Patna). It utilizes a fine-tuned Large Language Model (LLM) to provide relevant answers based on a custom dataset curated from IIT Patna-specific information[1][5].

The system is built using the Unsloth framework for efficient fine-tuning of the `unsloth/Llama-3.2-1B-Instruct` model[2][4][5]. The fine-tuning process involves two key stages:
1.  **Pre-training Phase:** The model is first adapted to understand the specific context of IIT Patna by training on relevant text data gathered from official sources and news articles[4][5].
2.  **Instruction Tuning Phase:** The pre-trained model is then further fine-tuned on a dataset of question-answer pairs related to IIT Patna life, enabling it to respond accurately to user queries[2][5].

This repository contains the scripts for data preparation, model training (both pre-training and instruction tuning), and evaluation.

## Problem Statement

The goal, as outlined in the Group 19 project description, is to design and implement a question-answering system focused on IIT Patna student life. This involves creating a custom dataset containing questions and answers about various aspects like fests, academics, attendance policies, marking schemes, course details, and more, and then using this dataset to train a model capable of answering user questions accurately[1][5].

## Pipeline / Methodology

The project follows these steps:

1.  **Context Generation:** Relevant information about IIT Patna was collected by scraping the official IIT Patna website and various news articles. The focus was on gathering context related to fests, academic life, placement scenarios, course details, attendance policies, and cultural events[1][5]. This context is compiled in `data/context_new.csv`.
2.  **Q&A Pair Generation:** Google's GenAI Python library with the Gemini 2.0 Flash Lite model was used to automatically generate descriptive question-answer pairs based on the collected context[5]. These pairs form the instruction-tuning dataset (`data/final_qa_dataset_train.csv` and `data/final_qa_dataset_test.csv`).
3.  **Model Fine-tuning:**
    *   **Pre-training:** The base LLaMA 3.2-1B Instruct model is first pre-trained on the collected IIT Patna context (`data/context_new.csv`) using the `scripts/pretrain_iitp.py` script. This step helps the model learn the domain-specific vocabulary and nuances[4][5]. LoRA (Low-Rank Adaptation) is employed via Unsloth for efficient adaptation[4][5].
    *   **Instruction Tuning:** The model (ideally, the one resulting from the pre-training phase) is then instruction-tuned using the generated Q&A pairs (`data/final_qa_dataset_train.csv`) with the `scripts/instruction_tune.py` script. This phase teaches the model to follow instructions and answer questions based on the learned context[2][5]. LoRA is used again for parameter-efficient tuning[2][5]. A specific prompt format is used to structure the input for the model[2].
4.  **Evaluation:** The performance of the fine-tuned model is evaluated on a held-out test set (`data/final_qa_dataset_test.csv`) using the `scripts/test_pretrain.py` script[3][5]. Standard NLP metrics such as BLEU, ROUGE, and BERTScore F1 are calculated to assess the quality of the generated answers[5].

## Dataset

The project utilizes three main data files located in the `data/` directory:

*   `context_new.csv`: Contains the contextual text data scraped from IIT Patna sources, used for the pre-training phase[4].
*   `final_qa_dataset_train.csv`: Contains the question-answer pairs generated using Gemini, used for the instruction-tuning phase[2].
*   `final_qa_dataset_test.csv`: Contains question-answer pairs held out for evaluating the final model's performance[3].

The Q&A datasets cover topics such as academics, fests, attendance structure, marking schemes, specific course details, placements, and cultural events at IIT Patna[1][5].

## Repository Structure

## Setup and Installation

1.  **Clone the repository:**
    ```
    git clone <your-repository-link>
    cd <repository-name>
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    Ensure you have CUDA installed for GPU acceleration if available. Install the required Python packages using pip:
    ```
    pip install -r requirements.txt
    ```
    Key dependencies include `unsloth`, `torch`, `transformers`, `datasets`, `pandas`, `trl`, `accelerate`[2][3][4].

## Usage

**1. Pre-training Phase:**

*   Run the pre-training script using the context data:
    ```
    python scripts/pretrain_iitp.py
    ```
*   This script uses `data/context_new.csv` as input[4].
*   Checkpoints will be saved in the `outputs_pretrain/` directory[4].

**2. Instruction Tuning Phase:**

*   Modify `scripts/instruction_tune.py` to load the desired pre-trained checkpoint if applicable (e.g., uncomment and set `model_name = "./outputs_pretrain/checkpoint-<X>"`)[2]. Currently, it loads the base model `unsloth/Llama-3.2-1B-Instruct`[2].
*   Run the instruction tuning script:
    ```
    python scripts/instruction_tune.py
    ```
*   This script uses `data/final_qa_dataset_train.csv` as input[2].
*   Fine-tuned model checkpoints will be saved in the `outputs_instruction_tuned_only/` directory[2].

**3. Testing / Inference:**

*   Modify `scripts/test_pretrain.py` to load the path to your final instruction-tuned checkpoint (e.g., `model_name = "./outputs_instruction_tuned_only/checkpoint-<Y>"`) [3].
*   Run the testing script:
    ```
    python scripts/test_pretrain.py
    ```
*   This script uses `data/final_qa_dataset_test.csv` as input and generates answers[3].
*   The generated outputs will be saved to `generated_outputs/generated_outputs_test_baseline_only_pt.csv`[3].

## Model Details

*   **Base Model:** `unsloth/Llama-3.2-1B-Instruct`[2][4]
*   **Fine-tuning Framework:** Unsloth[2][4][5]
*   **Techniques:**
    *   LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning[2][4][5].
    *   4-bit quantization (optional, enabled via `load_in_4bit = True` flag in scripts) for reduced memory usage[2][3][4].
    *   Gradient checkpointing for memory efficiency during training[2][4].

## Evaluation Results

The model's performance was evaluated using BLEU, ROUGE (ROUGE-1, ROUGE-2, ROUGE-L), and BERTScore (F1). The results indicate that the combined approach of pre-training followed by instruction tuning yields significantly better performance compared to only pre-training or only instruction tuning[5].

**Performance Comparison:**

| Metric            | Only Pre-training | Only Instruction Tuning (Baseline) | Pre-training + Instruction Tuning |
| :---------------- | :---------------: | :--------------------------------: | :-------------------------------: |
| BLEU Score        |      0.0928       |              0.1011              |             **0.3085**            |
| ROUGE-1           |      0.2545       |              0.2181              |             **0.4570**            |
| ROUGE-2           |      0.1047       |              0.1227              |             **0.2863**            |
| ROUGE-L           |      0.2123       |              0.1908              |             **0.4119**            |
| BERTScore (F1)    |      0.4986       |              0.5465              |             **0.9216**            |

