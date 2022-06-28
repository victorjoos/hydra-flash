from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def resolver(root, file_id):
    return Path(root) / f"{file_id}.jpg"


def show_predictions(images, predictions, show, save):
    for i, (image, pred) in enumerate(zip(images, predictions)):
        image = image["input"]
        if hasattr(image, "shape"):
            image = image.permute(1, 2, 0)
        fig = plt.figure()
        plt.imshow(image)
        if isinstance(pred, str):
            plt.title(f"{pred}")
        elif isinstance(pred, list):
            if np.array(pred).ndim == 2:
                plt.imshow(pred, cmap="tab20", alpha=0.5)
            else:
                plt.title(f"{pred}")

        if save:
            plt.savefig(f"{i}.png")
        if show:
            plt.show()
