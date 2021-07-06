#!/usr/bin/env python

import sys

import numpy as np
import typer
import matplotlib.pyplot as plt
from itertools import islice
import webdataset as wds
from webdataset.iterators import getfirst
from . import degrade as ocrodeg
import scipy.ndimage as ndi
import random
from dataclasses import dataclass


app = typer.Typer()


@dataclass
class Degrade:
    blur0: float = 0.5
    blur1: float = 3.0
    distortsigma0: float = 0.1
    distortsigma1: float = 10.0
    distort0: float = 0.5
    distort1: float = 5.0

    def distort(self, image):
        sigma = random.uniform(self.distortsigma0, self.distortsigma1)
        delta = random.uniform(self.distort0, self.distort1)
        noise = ocrodeg.bounded_gaussian_noise(image.shape, sigma, delta)
        return ocrodeg.distort_with_noise(image, noise)

    def blur(self, image):
        sigma = random.uniform(self.blur0, self.blur1)
        return ndi.gaussian_filter(image, sigma)

    def degrade(self, image):
        result = {}
        distorted = self.distort(image)
        blurred = self.blur(distorted)
        pms = ocrodeg.printlike_multiscale(blurred, blotches=1e-7)
        pfs = ocrodeg.printlike_fibrous(blurred, blotches=1e-7)
        result["dst.jpg"] = distorted
        result["blr.jpg"] = blurred
        result["pms.jpg"] = np.amax(pms) - pms
        result["pfs.jpg"] = np.amax(pfs) - pfs
        self.result = result
        return result


@app.command()
def degrade(
    fname: str,
    extensions: str = "jpg;jpeg;png;page.jpg;page.png;bin.jpg;bin.png",
    display: int = -1,
    maxrec: int = 999999999,
    output: str = None,
    options: str = "",
):
    """Binarize a shard of images."""
    ds = wds.WebDataset(fname).decode("l8")
    sink = wds.TarWriter(output)
    count = 0
    degrade = Degrade()
    options = options.split(",")
    options = [x.split("=") for x in options if x != ""]
    for k, v in options:
        setattr(degrade, k, eval(v))
    for sample in islice(ds, 0, maxrec):
        key = sample["__key__"]
        image = getfirst(sample, extensions, None, False)
        if image is None:
            print(f"{count}/{maxrec} {key} MISSING", file=sys.stderr)
        print(f"{count}/{maxrec} {key}", file=sys.stderr)
        assert image.dtype == np.uint8
        image = image / 255.0
        if image.ndim == 3:
            image = np.mean(image, 2)

        result = dict(sample)
        extra = degrade.degrade(image)
        result.update(extra)

        sink.write(result)

        if display > 0 and count % display == 0:
            plt.ion()
            plt.clf()
            plt.subplot(231)
            plt.imshow(image, cmap="gray")
            plt.subplot(232)
            plt.imshow(result["pms.jpg"], cmap="gray")
            plt.subplot(233)
            plt.imshow(result["pfs.jpg"], cmap="gray")
            plt.subplot(234)
            plt.imshow(result["dst.jpg"], cmap="gray")
            plt.subplot(235)
            plt.imshow(result["blr.jpg"], cmap="gray")
        if display > 0:
            plt.ginput(1, 0.02)
        count += 1
    sink.close()


if __name__ == "__main__":
    app()
