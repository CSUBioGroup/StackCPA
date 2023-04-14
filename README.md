# StackCPA

**StackCPA** is a stacking model for compound-protein binding affinity prediction based on pocket multi-scale features

## Installation
In order to get StackCPA, you need to clone this repo.

```
 Git clone https://github.com/CSUBioGroup/StackCPA/
 cd stackcpa
```
The easiest way to prepre the environment with all required packages is to use conda:

` conda env create -f environment.yml `

Remember to activate your environment before running the scripts:

` source activate stackcpa `

To prepare the protein and compound representations, please follow the instructions of **mol2vec** and **graph2vec** and finished the installation.

```
  cd ./code
  pip install git+https://github.com/samoturk/mol2vec
  pip install git+https://github.com/benedekrozemberczki/graph2vec.git
```
Now you are ready to use StackCPA.

## Usage
### Predict the affinity for a compound-protein pair
1.	Put `.pdb` files of protein and compound under the directory `./datasets`.

2.	Follow `data_processd.ipynb` to prepare `.npy` data for protein and compound for further prediction.

3.	We here provide 3 models trained on 3 datasets respectively for you to choose, which is under the directory `~/StackCPA/saved_model/`. 
    
    >Take the `pdbbind_save_model.pkl` as the example:
    
    ```
    # Run the commandline
    predict.py -model ./saved_model/pdb_save_model.pkl -data ./data/*.npy
    ```
### Train a new model
Follow the process of data preparation in the **Predict the affinity for a compound-protein pair (step 1 and 2)**,
 
 and then train a new model for your own data. The model will be saved under the directory `./saved_model/` .
 
 ```
    # Run the commandline
    train.py --train_data ./data/*.npy --train_label ./data/*.npy 
 ```
 ### Specify hypermeters for base models
 There are many hyperparameters of the base models (i.g. learning rate, number of estimators, number of leaves, lambda). For more information, see source code of train.py.
 
 ```
    # Run the commandline
    train.py --train_data ./data/*.npy --train_label ./data/*.npy --c_lr 0.01 --l_lr 0.1 --x_lr 0.007
 ```

## Requirements
In order to use the provided model and run all scripts you need:

rdkit==2022.3.5

mol2vec==0.1

r-protr==1.6_2

pytorch==1.10.2  

scipy==1.7.3

scikit-learn==1.0.2 

xgboost==1.4.2  

lightgbm==3.3.2

catboost==1.0.4  

pandas==1.3.5

numpy==1.21.6

## References
PDBbind database：http://www.pdbbind.org.cn/

KIBA：https://researchportal.helsinki.fi/fi/datasets/kiba-a-benchmark-dataset-for-drug-target-prediction
