# The Folk music transformer

## Setup environment

Create new environment with

```bash
conda create --file environment.yml
```

Activate with 

```bash
conda activate magenta 
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

