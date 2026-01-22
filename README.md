# Salesforce CausalAI Library
# Salesforce CausalAI Library

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Conda](https://img.shields.io/badge/Conda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Ray](https://img.shields.io/badge/Ray-F76C03?style=for-the-badge&logo=ray&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## Installation and Quick Start

```bash
# Create conda environment and install the library
conda create -n causal_ai_env python=3.9
conda activate causal_ai_env
git clone https://github.com/salesforce/causalai.git
cd causalai
pip install .

# Run example causal discovery and inference code (Python inline script)
python -c "
from causalai.models.tabular.pc import PC
from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
from causalai.data.data_generator import DataGenerator
from causalai.models.common.prior_knowledge import PriorKnowledge
from causalai.data.tabular import TabularData
from causalai.data.transforms.time_series import StandardizeTransform
from causalai.models.tabular.causal_inference import CausalInference
from sklearn.linear_model import LinearRegression
import numpy as np
from causalai.misc.misc import get_precision_recall

# Define SEM for causal discovery
fn = lambda x: x
coef = 0.1
sem = {
    'a': [], 
    'b': [('a', coef, fn), ('f', coef, fn)], 
    'c': [('b', coef, fn), ('f', coef, fn)],
    'd': [('b', coef, fn), ('g', coef, fn)],
    'e': [('f', coef, fn)], 
    'f': [],
    'g': [],
}
T = 5000
data_array, var_names, graph_gt = DataGenerator(sem, T=T, seed=0, discrete=False)

standardizer = StandardizeTransform()
standardizer.fit(data_array)
data_trans = standardizer.transform(data_array)
data_obj = TabularData(data_trans, var_names=var_names)

prior_knowledge = PriorKnowledge(forbidden_links={'a': ['b']})
ci_test = PartialCorrelation()
pc = PC(data=data_obj, prior_knowledge=prior_knowledge, CI_test=ci_test, use_multiprocessing=False)
result = pc.run(pvalue_thres=0.01, max_condition_set_size=2)

graph_est = {node: [] for node in result.keys()}
for node, res in result.items():
    graph_est[node].extend(res['parents'])
    print(f'{node}: {res[\"parents\"]}')

precision, recall, f1_score = get_precision_recall(graph_est, graph_gt)
print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}')

# SEM for causal inference
coef = 0.5
sem = {
    'a': [], 
    'b': [('a', coef, fn), ('f', coef, fn)], 
    'c': [('b', coef, fn), ('f', coef, fn)],
    'd': [('b', coef, fn), ('g', coef, fn)],
    'e': [('f', coef, fn)], 
    'f': [],
    'g': [],
}
data, var_names, graph_gt = DataGenerator(sem, T=T, seed=0, discrete=False)

def create_treatment(name, treat_val, control_val):
    return dict(var_name=name, treatment_value=treat_val, control_value=control_val)

t1, t2 = 'a', 'b'
target = 'c'
target_index = var_names.index(target)

intervention1_treat = 100 * np.ones(T)
intervention2_treat = 10 * np.ones(T)
intervention1_control = 0 * np.ones(T)
intervention2_control = -2 * np.ones(T)

treatments = [
    create_treatment(t1, intervention1_treat, intervention1_control),
    create_treatment(t2, intervention2_treat, intervention2_control),
]

causal_inference = CausalInference(data, var_names, graph_gt, LinearRegression, discrete=False, method='causal_path')
ate, _, _ = causal_inference.ate(target, treatments)
print(f'Estimated ATE (causal_path): {ate:.2f}')

causal_inference = CausalInference(data, var_names, graph_gt, LinearRegression, discrete=False, method='backdoor')
ate, _, _ = causal_inference.ate(target, treatments)
print(f'Estimated ATE (backdoor): {ate:.2f}')

intervention_data1, _, _ = DataGenerator(sem, T=T, seed=0, intervention={t1: intervention1_treat, t2: intervention2_treat})
intervention_data2, _, _ = DataGenerator(sem, T=T, seed=0, intervention={t1: intervention1_control, t2: intervention2_control})
true_ate = (intervention_data1[:, target_index] - intervention_data2[:, target_index]).mean()
print(f'True ATE: {true_ate:.2f}')
"

# Launch and terminate UI
./launch_ui.sh
./exit_ui.sh

# BibTeX citation
echo '@article{salesforce_causalai23,
  title={Salesforce CausalAI Library: A Fast and Scalable framework for Causal Analysis of Time Series and Tabular Data},
  author={Arpit, Devansh and Fernandez, Matthew and Feigenbaum, Itai and Yao, Weiran and Liu, Chenghao and Yang, Wenzhuo and Josel, Paul and Heinecke, Shelby and Hu, Eric and Wang, Huan and Hoi, Stephen and Xiong, Caiming and Zhang, Kun and Niebles, Juan Carlos},
  year={2023},
  eprint={arXiv preprint arXiv:2301.10859},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}' > citation.bib


🔥 **Latest Updates** 🔥  
- Introduced GES, LINGAM, and GIN algorithms for causal discovery on tabular data  
- Added Grow-Shrink algorithm for Markov Blanket discovery on tabular datasets  
- Extended support for heterogeneous datasets combining discrete and continuous variables  
- Developed benchmarking modules for tabular and time series data to evaluate causal discovery methods against challenges like graph sparsity, sample size, noise type, and signal-to-noise ratio  
- Added Root Cause Analysis capabilities for tabular and time series data  

---

## Overview
![Image Alt](https://github.com/salesforce/causalai/blob/e03c9d8e0afcdf8f3b9681f6140051327eaa3a71/assets/causalai_pipeline.png)

Salesforce CausalAI is an open-source Python library designed for causal analysis using observational data. It supports both causal discovery and inference across tabular and time series datasets of various types — discrete, continuous, and mixed. The library offers a flexible set of algorithms that can capture both linear and non-linear causal relationships, and leverages multiprocessing to improve performance on large datasets. A synthetic data generator helps users create controlled datasets with known causal structures to evaluate algorithms effectively. Furthermore, CausalAI includes benchmarking tools to compare different algorithms across datasets with varying complexity and challenges. For ease of use, a code-free user interface allows users to perform causal analyses without writing any code.

---

## Problems Addressed

Traditional causal analysis with observational data presents several challenges:  
- Managing different data types and missing values  
- Handling the complexity of both linear and non-linear causal relationships  
- Working with time series and tabular formats seamlessly  
- Running computationally expensive causal discovery and inference algorithms efficiently  
- Benchmarking and evaluating algorithm performance under diverse data challenges  
- Providing accessible interfaces for users unfamiliar with coding  

CausalAI addresses these pain points by providing a unified, scalable framework with flexible algorithms, synthetic data support, benchmarking modules, and a user-friendly UI.
![Image Alt](https://github.com/salesforce/causalai/blob/e03c9d8e0afcdf8f3b9681f6140051327eaa3a71/assets/causalai_comparison.png)


---

## Solutions Provided

CausalAI offers the following:  
- **Comprehensive algorithms** for causal discovery, inference, Markov Blanket discovery, and Root Cause Analysis  
- **Multi-data support:** works with tabular and time series data of discrete, continuous, and mixed types  
- **Missing data handling:** robust to NaN and missing values  
- **Synthetic data generator:** create data with defined structural causal models to test and benchmark methods  
- **Distributed computing:** optional multiprocessing support via Python’s Ray for faster computation on large datasets  
- **Targeted discovery:** focus on causal parents of specific variables to reduce computational load  
- **Visualization tools:** graphical representation of discovered causal graphs  
- **Integration of prior knowledge:** incorporate user-defined constraints and partial graph information  
- **Benchmarking modules:** evaluate algorithm accuracy and robustness across multiple data scenarios  
- **Code-free UI:** perform causal analyses through a web interface without writing code  

---

## Causal Discovery

CausalAI implements several causal discovery algorithms with different assumptions about hidden variables, data types, and noise models. For continuous data, algorithms like PC and Grow-Shrink support both linear and non-linear relationships depending on the conditional independence tests used, while other algorithms support linear relationships only.
![Image Alt](https://github.com/salesforce/causalai/blob/e03c9d8e0afcdf8f3b9681f6140051327eaa3a71/assets/cd_algos.png)

---

## Causal Inference

Supports estimation of causal effects from observational data:  
- **Average Treatment Effect (ATE):** measures expected outcome difference under treatment versus control interventions  
- **Conditional Average Treatment Effect (CATE):** ATE conditioned on other covariates  
- **Counterfactual inference:** estimates effect of interventions on specific samples while holding other variables fixed  

Users can choose linear or non-linear models depending on the underlying data relationship.

---

# Installation
conda create -n causal_ai_env python=3.9
conda activate causal_ai_env
git clone https://github.com/salesforce/causalai.git
cd causalai
pip install .

# Python example for causal discovery and inference
python -c "
from causalai.models.tabular.pc import PC
from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
from causalai.data.data_generator import DataGenerator
from causalai.models.common.prior_knowledge import PriorKnowledge
from causalai.data.tabular import TabularData
from causalai.data.transforms.time_series import StandardizeTransform
from causalai.models.tabular.causal_inference import CausalInference
from sklearn.linear_model import LinearRegression
import numpy as np
from causalai.misc.misc import get_precision_recall

# SEM setup for discovery
fn = lambda x: x
coef = 0.1
sem = {
    'a': [], 
    'b': [('a', coef, fn), ('f', coef, fn)], 
    'c': [('b', coef, fn), ('f', coef, fn)],
    'd': [('b', coef, fn), ('g', coef, fn)],
    'e': [('f', coef, fn)], 
    'f': [],
    'g': [],
}
T = 5000
data_array, var_names, graph_gt = DataGenerator(sem, T=T, seed=0, discrete=False)

standardizer = StandardizeTransform()
standardizer.fit(data_array)
data_trans = standardizer.transform(data_array)
data_obj = TabularData(data_trans, var_names=var_names)

prior_knowledge = PriorKnowledge(forbidden_links={'a': ['b']})
ci_test = PartialCorrelation()
pc = PC(data=data_obj, prior_knowledge=prior_knowledge, CI_test=ci_test, use_multiprocessing=False)
result = pc.run(pvalue_thres=0.01, max_condition_set_size=2)

graph_est = {node: [] for node in result.keys()}
for node, res in result.items():
    graph_est[node].extend(res['parents'])
    print(f'{node}: {res[\"parents\"]}')

precision, recall, f1_score = get_precision_recall(graph_est, graph_gt)
print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}')

# SEM setup for causal inference
coef = 0.5
sem = {
    'a': [], 
    'b': [('a', coef, fn), ('f', coef, fn)], 
    'c': [('b', coef, fn), ('f', coef, fn)],
    'd': [('b', coef, fn), ('g', coef, fn)],
    'e': [('f', coef, fn)], 
    'f': [],
    'g': [],
}
data, var_names, graph_gt = DataGenerator(sem, T=T, seed=0, discrete=False)

def create_treatment(name, treat_val, control_val):
    return dict(var_name=name, treatment_value=treat_val, control_value=control_val)

t1, t2 = 'a', 'b'
target = 'c'
target_index = var_names.index(target)

intervention1_treat = 100 * np.ones(T)
intervention2_treat = 10 * np.ones(T)
intervention1_control = 0 * np.ones(T)
intervention2_control = -2 * np.ones(T)

treatments = [
    create_treatment(t1, intervention1_treat, intervention1_control),
    create_treatment(t2, intervention2_treat, intervention2_control),
]

causal_inference = CausalInference(data, var_names, graph_gt, LinearRegression, discrete=False, method='causal_path')
ate, _, _ = causal_inference.ate(target, treatments)
print(f'Estimated ATE (causal_path): {ate:.2f}')

causal_inference = CausalInference(data, var_names, graph_gt, LinearRegression, discrete=False, method='backdoor')
ate, _, _ = causal_inference.ate(target, treatments)
print(f'Estimated ATE (backdoor): {ate:.2f}')

intervention_data1, _, _ = DataGenerator(sem, T=T, seed=0, intervention={t1: intervention1_treat, t2: intervention2_treat})
intervention_data2, _, _ = DataGenerator(sem, T=T, seed=0, intervention={t1: intervention1_control, t2: intervention2_control})
true_ate = (intervention_data1[:, target_index] - intervention_data2[:, target_index]).mean()
print(f'True ATE: {true_ate:.2f}')
"

# Launch and exit UI
./launch_ui.sh
./exit_ui.sh

# BibTeX citation
echo '@article{salesforce_causalai23,
  title={Salesforce CausalAI Library: A Fast and Scalable framework for Causal Analysis of Time Series and Tabular Data},
  author={Arpit, Devansh and Fernandez, Matthew and Feigenbaum, Itai and Yao, Weiran and Liu, Chenghao and Yang, Wenzhuo and Josel, Paul and Heinecke, Shelby and Hu, Eric and Wang, Huan and Hoi, Stephen and Xiong, Caiming and Zhang, Kun and Niebles, Juan Carlos},
  year={2023},
  eprint={arXiv preprint arXiv:2301.10859},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}' > citation.bib
