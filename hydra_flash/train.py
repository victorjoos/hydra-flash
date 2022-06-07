import hydra
from hydra.utils import call, instantiate as hydra_instantiate
from functools import partial
import logging

log = logging.getLogger(__name__)

instantiate = partial(hydra_instantiate, _convert_="all")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # Preprocess : download and/or unzip data
    call(cfg.preprocess)

    datamodule = instantiate(cfg.datamodule)

    model = instantiate(cfg.model, labels=datamodule.labels)

    trainer = instantiate(cfg.trainer)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    predict_datamodule = instantiate(cfg.predict_datamodule)
    predictions = trainer.predict(model, datamodule=predict_datamodule, output="labels")
    log.info(f"predictions : {predictions}")
