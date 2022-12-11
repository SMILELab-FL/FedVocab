<h1 align="center">  
    <p> FedeVocab </p>  
</h1>  
<p align="center"> 
	 <img alt="Licence" src="https://img.shields.io/badge/License-Apache%202.0-yellow">
	 <img alt="Build" src="https://img.shields.io/badge/build-processing-green">
 </p>
Code for paper "Federated Model Decomposition with Private Vocabulary for Text Classification".  In this paper, we propose a <u>fe</u>drated model <u>de</u>composition method that protects the privacy of <u>vocab</u>ularies, shorted as FedeVocab.

## Installation
### Directory Structure
Before using our code, we recommend running FedeVocab according to the following directory structure:
```grapha  
├── workspace  
│   └── data   
|   |   └── fednlp_data  
│   ├── pretrain
│   │   └── cv  
│   ├── output  
│   └── code  
│       └── fedevocab
``` 
You can run the following command:
```bash  
mkdir workspace  
cd workspace  
mkdir data  
mkdir code  
mkdir pretrained  
cd pretrained  
mkdir nlp  
cd ..  
cd code  
``` 
### Requirement
The `python` version of the running environment is `3.7+` and the `pytorch` version is `1.10+`.
```bash  
git clone 
cd fedevocab  
pip install -r resquirements.txt  
```

## How to run FedeVocab
Our code is built on FedNLP. To use our code, you must clone FedNLP.
### Utility Experiments
```bash  
bash run/detlm_alone/fedrun_sweep.sh sst_2 distilbert v100 2
```
### Privacy Experiments
```bash 
bash run/attack/dlg.sh /workspace {save_times} {model_type} {gpu_id} {batch_size}
```



## Licence
[Apache 2.0](./LICENSE)
