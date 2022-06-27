# Hydra tutorial

This repository contains a very simple network training using pytorch lightning flash.
The configuration management is done using Hydra (https://hydra.cc/), as the goal
of this code is to showcase how Hydra can be used.

## Installing

You can install this code with poetry (install poetry at : https://python-poetry.org/docs/#installation) using : 

```
poetry install
```

## Running

Once the project has been installed, you can run it with the following command:

```
poetry run flashtrain
```

You can, for example, change the learning rate using : 

```
poetry run flashtrain model.learning_rate=1e-2
```