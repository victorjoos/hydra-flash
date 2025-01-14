import hydra
from hydra.utils import call, instantiate as hydra_instantiate
from functools import partial
from omegaconf import OmegaConf
import logging

from hydra_flash.utils import show_predictions

log = logging.getLogger(__name__)

instantiate = partial(hydra_instantiate, _convert_="all")


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    # Preprocess : download and/or unzip data
    call(cfg.preprocess)

    # Create datamodule from downloaded files
    datamodule = instantiate(cfg.datamodule)

    # Create model
    model_kwargs = {}
    for kw in cfg.model.complete:
        model_kwargs[kw] = getattr(datamodule, kw)
    del cfg.model.complete
    model = instantiate(cfg.model, **model_kwargs)

    # Finetune model
    trainer = instantiate(cfg.trainer)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # Predict and show predictions
    predict_datamodule = instantiate(cfg.predict_datamodule)
    predictions = trainer.predict(model, datamodule=predict_datamodule, output="labels")

    show_predictions(
        predict_datamodule.predict_dataset, predictions[0], cfg.show, cfg.save
    )


if __name__ == "__main__":
    main()
