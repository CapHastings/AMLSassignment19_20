# README
This is a brief instruction on AMLSassignment19_20. 
## Brief Description

The project includes four tasks:

- A1: Gender detection on celeba dataset.
- A2: Emotion detection on celeba dataset.
- B1: Face shape recognition on cartoon_set dataset.
- B2: Eye color recognition on cartoon_set dataset.

codes of each task are placed in A1, A2, B1 and B2 folder respectively and corresponding datasets are also stored in Datasets folder. 

## Directory Tree

The directory tree is presented below: 

```
├── A1
│   ├── a1.py
│   ├── data_preprocess.py
│   └── model_tuning.py
├── A2
│   ├── a2.py
│   ├── data_preprocess.py
│   ├── model_tuning.py
│   └── pre-trained_model.h5
├── B1
│   ├── b1.py
│   ├── data_preprocess.py
│   └── model_tuning.py
├── B2
│   ├── b2.py
│   ├── data_preprocess.py
│   └── model_tuning.py
├── Datasets
│   ├── cartoon_set
│   ├── cartoon_set_b1
│   ├── cartoon_set_b2
│   ├── celeba
│   ├── celeba_set_a1
│   └── celeba_set_a2
├── README.md
├── main.py
```

each python file with specific purposes is placed orderly, and python files with the same (similar) name in different tasks have similar functions. To conclude: 

- ```data_preprocess.py```: process data before being fit by a model, including basic functions of image preprocessing, data augmentation, the loading and export of data, etc. 
- ```model_tuning.py```: include functions of fine-tuning the selected model via cross-validation, the train/test accuracy calculation.
- ```a1.py (a2. py, etc.)```: include a class whose objects can invoke the functions of above two python files.
- ```pre-trained_model.h5```: a pre-trained CNN model.
- ```main.py```: integrate four tasks in one—load datasets and train the models of four tasks, and print the train/test accuracy respectively.

## Usage

**Warm Reminder:** The project interpreter is Python 3.7, so errors may occur in Python 2.x. The relative path is written as macOS path, please make minor amendments if you are using Windows. The location of datasets and python files should be entirely identical to the directory tree.

1. Download datasets from [here](https://liveuclac-my.sharepoint.com/:u:/g/personal/uceepz0_ucl_ac_uk/EdrbNR_YLr1PqeIpdF8tpXwBE67HR82zUwEp6-sNkpsYug?e=luLmWP), unzip and move the whole folder "Datasets" to the location shown in the directory tree.
2. Change your current working directory to the project in Terminal: ```$ cd /AMLSassignment19_20-master```	
3. Run ```$ python main.py```

If you cannot run it successfully after many tries, feel free to contact me by email.

## Required Packages

The project requires the installation of following packages and their dependencies, do check before a run.

- Numpy 1.16.2
- Pandas 0.23.4 
- Scikit-learn
- Keras 2.3.1
- Tensorflow
- PIL
- Imgaug 0.3.0 

## Contact

Email: pei.zhang.19@ucl.ac.uk
