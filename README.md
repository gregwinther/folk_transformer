# The Folk music transformer

## Setup environment

Create new environment with

```bash
conda create --file environment.yml
```

Activate with 

```bash
conda activate transformer 
```

Deactivate with

```bash
conda deactivate
```

## Google cloud storage SDK

You will need some stuff stored on the google cloud. Fetch this with `gsutil`
in the command line. This command fetches the pre-trained music transformer
models and some sample primers,

```bash
mkdir models
gsutil -q -m cp -r 'gs://magentadata/models/music_transformer/*' './models/'
```


## Run pre-trained model

Example run from command line;

```bash
python unconditional_sample.py -model_path='./models/checkpoints/unconditional_model_16.ckpt' -output_dir=./tmp -decode_length=1024 -primer_path='./models/primers/fur_elise.mid'
```

Also works without a primer.
