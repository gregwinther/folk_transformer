# Americana Folk music transformer

This project is part of the final delivery in a course on 
AI models at the University of Oslo.

We have created a model that was initialized from the
[MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset and applied transfer
learning by training it furter on Americana MIDI files. These files are not public
domain, regrettably. We have uploaded our model to [Zenodo](https://zenodo.org/record/4270422).

## Setup/Installation

We have relied on the `musicautobot` by [bearpelican](https://github.com/bearpelican/musicautobot).
Clone our modified version;

```bash
git clone https://github.com/gregwinther/musicautobot.git
```

Setup conda environment

```bash
conda env create -f ./musicautobot/environment.yml
```

Then activate environment

```bash
conda activate folk_transformer
```

## Download and generate music from pre-trained model.

Download our pretrained model from Zenodo;
```bash
wget https://zenodo.org/record/4270422/files/transfer_model.pth
```

If everythin installed correctly, you should be able to generate music as 
shown in the jupyter notebook `Generate.ipynb`.

## Train new model

In order to train a new model, first you need to gather a folder with MIDI files.
Use the script `preprocess_data.py` to turn this into a processed, tokenized numpy object.
Then you _should_ be able to run the trainer, as exemplified in the script `transfer_train.py`.
This script shows how we did transfer training. You can train a model from scratch by 
not including the `pretrained_path`.
