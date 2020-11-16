# Eye movement classification with TCNs

Temporal Convolutional Networks for the tertiary eye movement classification problem (3EMCP)

-----

This repository provides access to the codebase, models, and evaluation results described in the paper *Eye Movement Classification with Temporal Convolutional Networks* (link will be provided soon). Please note that most of the code shared here was originally implemented by [Startsev et al.](https://github.com/MikhailStartsev/deep_em_classifier). We only added support for TCNs, upgraded it to Python 3, implemented a few new tools, and a new feature extractor in Python (the original one was in MATLAB).



### Setting up

In order to train a new TCN-based model or evaluate the previously trained models, you need first to download some large compacted files containing all the necessary data and extract them according to the following instructions:
* Download [GazeCom_new_features.zip](https://drive.google.com/file/d/1mQRa0trH78zEC3lhmGt4qpyHoENY4-9E/view?usp=sharing), which contains the GazeCom pre-computed features (with extra scales and features), and extract it to ``data/inputs/``
* Download [models.zip](https://drive.google.com/file/d/1AUdZgpuOo1963PRtanmndGCcBVmJA3K8/view?usp=sharing), a compressed file with all trained models, and extract it to the repository root folder
* Download [outputs.zip](https://drive.google.com/file/d/1tC8Qj2Me8y6sgzAXrVQSlH210Ti7aBkg/view?usp=sharing), a file with the generated outputs of the trained models for evaluation, and extract it to the repository root folder



### Known dependencies

* Python 3.6+
* [keras-tcn](https://github.com/philipperemy/keras-tcn)
* TensorFlow 2.0+
* Numpy
* [liac-arff](https://github.com/renatopp/liac-arff)



### Training

To train a new TCN model, you should run the ``train_tcn.py`` script, but first you need to set up the training parameters. This is done in the code, managing the constants that are already filled with some inital values. They are supposed to be self-explanatory, but, if needed, a more thorough documentation can be found [here](https://github.com/MikhailStartsev/deep_em_classifier). 

In particular, pay attention to the ``MODEL_FOLDER`` and ``OUT_FOLDER`` constants, as they define where the trained model is going to be stored and the respective outputs generated.



### Evaluating

Evaluation can be done once there are generated outputs. We already provide the outputs for each trained model described in the paper in the ``outputs.zip`` file. Note that the evaluator script uses the ``sp_tool`` provided by  [Startsev et al.](https://github.com/MikhailStartsev/deep_em_classifier) And remember to check whether you are providing the correct output path for the constant ``OUT_FOLDER`` inside the evaluator. The scores will be stored at a *json* file within this folder. To run the evaluation, do:

```
$python3 evaluate.py generated_output_folder
```

  

### Filtering

The knowledge-based filter (*optional*) should be run after the training is complete but **before** evaluation. The filter script is located at ``feature_extraction/filter.py``. It works by processing an entire folder containing the generated outputs and exporting the filtered entries to a specified new path:

```
$python3 feature_extraction/filter.py entry_path output_path
```

  