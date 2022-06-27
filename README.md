# Hydra tutorial

This repository contains a very simple network training using pytorch lightning flash.
The configuration management is done using Hydra (https://hydra.cc/), as the goal
of this code is to showcase how Hydra can be used.

## Installing

In a separate virtual environement, install using `python -m pip install -r requirements.txt`.

If you're using poetry (https://python-poetry.org/docs/#installation), you can install using : 

```
poetry install
```

## Running

Once the project has been installed, you can run it with the following command:

```
python -m hydra_flash.train
```

You can, for example, change the learning rate using : 

```
python -m hydra_flash.train model.learning_rate=1e-2
```
