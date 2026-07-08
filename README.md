# Boosting with LLMs: A Hybrid LLM-Boosting System for Automated Feature Type Inference

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Directory and File Details](#directory-and-file-details)
  - [data/](#data)
    - [Benchmark-Labeled-Data/](#benchmark-labeled-data)
  - [resources/](#resources)
  - [batches/](#batches)
  - [logs/](#logs)
  - [figures/](#figures)
  - [utils/](#utils)
  - [results/](#results)
  - [Notebooks](#notebooks)
- [Contributing](#contributing)
- [License](#license)

## Overview
This repository contains the source code accompanying the paper _"Boosting with LLMs: A Hybrid LLM-Boosting System for Automated Feature Type Inference"_.

We first analyze the benchmark dataset and reproduce the results reported in the study [Towards Benchmarking Feature Type Inference for AutoML Platforms](https://dl.acm.org/doi/10.1145/3448016.3457274). Building on this foundation, we first investigate the use of LLMs directly for automated feature type inference by generating binary and multi-class predictions using zero-shot and few-shot prompting strategies. We then incorporate these LLM-generated predictions as additional features within a hybrid framework to further boost performance.

The proposed hybrid system combines semantic information extracted by LLMs with statistical characteristics derived from classical feature engineering approaches. The resulting feature representation is then used to train a machine learning model for feature type classification. In our experiments, [AutoGluon](https://auto.gluon.ai) with [XGBoost](https://xgboost.readthedocs.io) resulted in the best overall performance, where we achieved an average macro F1-score of 0.912 across 100 runs using only the LLM. The hybrid setup further improves performance to an average macro F1-score of 0.926, outperforming the reproduced baseline, which achieves an average macro F1-score of 0.852.

For additional details and a comprehensive discussion of the methodology and results, please refer to the paper.

## Project Structure

````
├── data/
│   ├── Benchmark-Labeled-Data/
│   │   ├── data_train.csv
│   │   ├── data_test.csv
│   │   ├── all_data.csv
├── resources/
│   ├── Dictionary/
│   │   ├── dictionaryName.pkl
│   │   ├── dictionarySample.pkl
├── batches/
│   ├── llm_feature_runner.py
│   ├── run_llm.sh
├── logs/
├── figures/
│   ├── CM_BestModel.pdf
│   ├── edit_distance.pdf
│   ├── total_inference_time_new_models.pdf
│   ├── violin_results_only_gpt.pdf
│   ├── original_feature_importances.pdf
│   ├── violin_differences.pdf
├── utils/
│   ├── feature_extraction.py
│   ├── feature_generator.py
│   ├── gpt.py
│   ├── hf_model.py
│   ├── metrics.py
│   ├── sortinghat_featurization.py
│   ├── splitting.py
├── results/
│   ├── featurized_data.csv
│   ├── featurized_data_with_llm_features.csv
│   ├── Autogluon_results_on_100_resamples.csv
│   ├── gpt_5_4.json
│   ├── gpt_5_4_fewshot.json
│   ├── qwen_3_5.json
│   ├── qwen_3_5_fewshot.json
│   ├── gemma_4.json
│   ├── gemma_4_fewshot.json
│   ├── gpt_5_4_tokens_zero-shot_prompts.csv
│   ├── gpt_5_4_tokens_few-shot_prompts.csv
├── AnalyzeBenchmarkTrainTestSplit.ipynb
├── EditDistances.ipynb
├── SortingHatFeatureImportances.ipynb
├── ReproduceSortingHatResults.ipynb
├── FindBestSplits.ipynb
├── CountGPTTokens.ipynb
├── RunGPTFeatures.ipynb
├── RunLocalModelFeatures.ipynb
├── EvaluateLLMFeaturesOnly.ipynb
├── HybridSystem.ipynb
├── seeds.txt
├── requirements.txt
├── requirements-local-models.txt
├── LICENSE
````

## How to Use

1. Before installing this project, make sure that Conda is installed on your system. If not, follow the official installation guide: https://docs.conda.io/en/latest/

2. This project requires **two separate Conda environments** with Python 3.11.11. This separation is necessary because different dependencies are used for the local models and the main experiments.
---

## Environment 1 — Main Experiments

1. Create the first Conda environment:
	```
	conda create -n boosting-with-llms python=3.11.11 -y
	```

2. Activate the environment:
	```
	conda activate boosting-with-llms
	```

3. Install all required dependencies:
	```
	pip install -r requirements.txt
	```
	
4. Set up the OpenAI API key for GPT experiments by creating an `.env` file in the project root directory. This step is required for running the notebooks and enabling API calls. Without this file, the `RunGPTFeatures.ipynb` notebook will not function as intended:  
	```
	OPEN_AI_KEY='<YOUR_OPEN_AI_API_KEY>'
	```

5. Install the SpaCy language model:
	```
	python -m spacy download en_core_web_lg
	```

6. Add the environment as a Jupyter kernel:
	```
	python -m ipykernel install --user --name=boosting-with-llms --display-name "Python 3.11 (boosting-with-llms)"
	```
---

## Environment 2 — Local Models & TensorFlow Data Validation

1. Create the second Conda environment:
	```
	conda create -n local-models python=3.11.11 -y
	```

2. Activate the environment:
	```
	conda activate local-models
	```

3. Install the required dependencies:
	```
	pip install -r requirements-local-models.txt
	```

4. Install the SpaCy language model:
	```
	python -m spacy download en_core_web_lg
	```

5. Add the environment as a Jupyter kernel:
	```
	python -m ipykernel install --user --name=boosting-with-llms-local
	python -m ipykernel install --user --name=local-models --display-name "Python 3.11 (local-models)
	```
---

## Running the Project

1. Start Jupyter Notebook from your terminal:
	```
	jupyter notebook
	```
2. Open the desired notebook in the browser.

3. Select the correct kernel via:
	```
	Kernel → Change Kernel
	```
Use:
- `boosting-with-llms` for main experiments
- `local-models` for local model experiments in notebook `RunLocalFeatures.ipynb` as well as the notebook `TFDV.ipynb` in folder `benchmark tools/`.

---

## Directory and File Details

### `data/`

#### `Benchmark-Labeled-Data/`

This folder contains the benchmark dataset and its train-test split provided in the original [SortingHat](https://dl.acm.org/doi/10.1145/3448016.3457274) paper:

-   **data_train.csv**: The training data.
-   **data_test.csv**: The test data.
-   **all_data.csv**: A combined version of both the training and test data.

### `resources/Dictionary/`

This directory contains pretrained models / vectorizers from the original [SortingHat](https://dl.acm.org/doi/10.1145/3448016.3457274) paper :


### `batches/`

This folder is used to store batch files generated during GPT-based experiments executed in `RunGPTFeatures.ipynb`. It is automatically populated once the notebook is run and contains the intermediate request/response data for GPT-based feature generation.


### `logs/`

This directory stores log files for LLM-based features. In particular, it records the runtime per prompt over all samples.


### `figures/`

This directory contains the visualizations generated from the notebooks, including plots and figures used in the paper.


### `utils/`

This directory contains utility scripts used across various notebooks:

- **feature_extraction.py**: Implements vectorization and word embedding methods for extracting features from attribute names.
- **feature_generator.py**: Builds the hybrid system by combining LLM-based features with classical statistical features.
- **gpt.py**: Provides an interface to the OpenAI API for sending prompts and retrieving GPT model outputs.
- **hf_model.py**: Loads and interfaces with local HuggingFace models for prompting and inference.
- **metrics.py**: Implements evaluation metrics, including confusion matrix computation and classification metrics.
- **sortinghat_featurization.py**: Reproduces the original [SortingHat](https://dl.acm.org/doi/10.1145/3448016.3457274) feature extraction approach, focusing on extracting statistical and regex-based feature representations.
- **splitting.py**: Implements stratified and group splitting functions for creating train-test splits.


### `results/`

This directory contains:

-   **featurized_data.csv**: A dataset that holds the core features with the original bi-gram features from SortingHat.
-   **featurized_data_with_llm_features.csv**: A dataset that holds the core features with the original bi-gram features from SortingHat and all additional features generated by the LLMs.
-   **Autogluon_results_on_100_resamples.csv**: The results from AutoGluon with different feature combinations on 100 resample runs.
-   **gpt_5_4.json**: Summarizes the performance of solely GPT-5.4 predictions in the zero-shot setting.
-   **gpt_5_4_fewshot.json**: Summarizes the performance of solely GPT-5.4 predictions in the few-shot setting.
-   **qwen_3_5.json**: Summarizes the performance of solely Qwen 3.5 9B predictions in the zero-shot setting.
-   **qwen_3_5_fewshot.json**: Summarizes the performance of solely Qwen 3.5 9B predictions in the few-shot setting.
-   **gemma_4.json**: Summarizes the performance of solely Gemma 4 E4B-it predictions in the zero-shot setting.
-   **gemma_4_fewshot.json**: Summarizes the performance of solely Gemma 4 E4B-it predictions in the few-shot setting.
-   **gpt_5_4_tokens_zero-shot_prompts.csv**: Contains minimum, average, and maximum token statistics from a single prompt in GPT-5.4 using zero-shot.
-   **gpt_5_4_tokens_few-shot_prompts.csv**: Contains minimum, average, and maximum token statistics from a single prompt in GPT-5.4 using few-shot.


### `seeds.txt`

This file contains the different random seeds used to split the dataset across resampling runs for reproducibility. The seeds were originally evaluated and selected in the notebook `FindBestSplits.ipynb`.


## Notebooks

-   **AnalyzeBenchmarkTrainTestSplit.ipynb**: This notebook analyzes the original train-test split provided in the [baseline](https://dl.acm.org/doi/10.1145/3448016.3457274).
    
-   **EditDistances.ipynb**: This notebook computes edit distances between features in the original train and test split to assess potential overfitting and data leakage effects.

-   **SortingHatFeatureImportances.ipynb**: This notebook analyzes feature importances of the pretrained vectorizer and model from SortingHat.

-   **ReproduceSortingHatResults.ipynb**: This notebook reproduces the baseline results by reimplementing the SortingHat pipeline using the original data splits and applying the pretrained model directly.

-   **FindBestSplits.ipynb**: This notebook identifies the best stratified and grouped train, validation, and test splits across our 100 resampling runs.
    
-   **CountGPTTokens.ipynb**: This notebook computes token usage statistics for a single prompt using GPT-5.4. Local models are excluded due to similar patterns.
    
-   **RunGPTFeatures.ipynb**: This notebook runs the GPT experiments. It first generates the batches and then executes them. It will result in an augmented version of the `featurized_data.csv` file, with additional GPT columns, which correspond to the predictions. It will save the augmented csv in the `results/` directory.

- **RunGPTFeatures.ipynb**: This notebook executes the GPT-based experiments. It first generates request batches and then processes them to obtain model predictions. The batches can either be executed directly within the notebook or by running the script `run_llm.sh` located in the `batches/` directory after batch creation. The resulting outputs are then used to generate an augmented version of `featurized_data.csv`, where additional columns contain the GPT-based predictions. The final enriched dataset is saved under `results/featurized_data_with_gpt_features.csv`.

-   **RunLocalModelFeatures.ipynb**: This notebook executes the local-LLM-based experiments. It downloads and loads the local LLM from HuggingFace and afterwards runs the prompts directly within the notebook. The resulting outputs are then used to generate an augmented version of `featurized_data.csv`, where additional columns contain the local-LLM-based predictions. The final enriched dataset is saved under `results/featurized_data_with_local_model_features.csv`.

- **EvaluateLLMFeaturesOnly.ipynb**: This notebook evaluates the LLM-based predictions in the standalone setting by directly comparing them to the ground truth. It is used to assess the performance of the LLM-only approach prior to integrating these predictions into the hybrid setup.

-   **HybridSystem.ipynb**: The main experiment notebook that performs a variety of experiments using different feature combinations in our hyrbid system. It includes AutoGluon as the classification model, where hyperparameter tuning through the 100 resampling runs is done automatically. The final results are stored in the `results/Autogluon_results_on_100_resamples.csv`.


## Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the Apache 2 License. See the LICENSE file for details.
