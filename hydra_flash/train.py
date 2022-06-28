from attr import has
import hydra
from hydra.utils import call, instantiate as hydra_instantiate
from functools import partial
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import logging

log = logging.getLogger(__name__)

instantiate = partial(hydra_instantiate, _convert_="all")


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    # Preprocess : download and/or unzip data
    call(cfg.preprocess)
    print(OmegaConf.to_container(cfg.datamodule, resolve=True))
    datamodule = instantiate(cfg.datamodule)

    model_kwargs = {}
    for kw in cfg.model.complete:
        model_kwargs[kw] = getattr(datamodule, kw)
    del cfg.model.complete
    model = instantiate(cfg.model, **model_kwargs)

    trainer = instantiate(cfg.trainer)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    predict_datamodule = instantiate(cfg.predict_datamodule)
    predictions = trainer.predict(model, datamodule=predict_datamodule, output="labels")

    for i, (image, pred) in enumerate(
        zip(predict_datamodule.predict_dataset, predictions[0])
    ):
        image = image["input"]
        if hasattr(image, "shape"):
            image = image.permute(1, 2, 0)
        fig = plt.figure()
        plt.imshow(image)
        if isinstance(pred, str) or isinstance(pred, list):
            plt.title(f"{pred}")
        else:
            plt.imshow(pred, cmap="tab20", alpha=0.5)

        if cfg.save:
            plt.savefig(f"{i}.png")
        if cfg.show:
            plt.show()


if __name__ == "__main__":
    main()
