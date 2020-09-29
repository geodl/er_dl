from pathlib import Path
from typing import Union, Optional

import numpy as np
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image

from project.config import cmap


PathLike = Union[str, Path]


def parse_filename(filename: PathLike, check_exists: bool = True, mkdir: bool = False) -> Path:
    assert isinstance(filename, (str, Path)), f'{filename} has unsupported type'
    filename = Path(filename).resolve()
    if check_exists:
        assert filename.exists(), f'File {filename} is not exists'
    if mkdir:
        if filename.is_dir():
            filename.mkdir(exist_ok=True, parents=True)
        elif filename.is_file():
            filename.parent.mkdir(exist_ok=True, parents=True)
        else:
            raise TypeError(f"{filename} cann't be interpreted as filename")
    return filename


def read_zondres2d_model(filename: PathLike):
    filename = parse_filename(filename)
    model = np.loadtxt(str(filename), usecols=(0, 1, 3), skiprows=0)
    return model


def show_zondres2d_model(filename: PathLike, plot: bool = False, min_value: Optional[float] = None, max_value: Optional[float] = None, return_as_pil: bool = False) -> plt.Figure:
    model = read_zondres2d_model(filename)
    x = np.sort(np.unique(model[:, 0]))
    z = np.sort(np.unique(model[:, 1]))
    rho = np.reshape(model[:, 2], (len(x), len(z)))
    rho = np.rot90(rho)

    rho = np.log(rho)

    if min_value is not None:
        min_rho = np.max((np.log(min_value), 0.0001))
    else:
        min_rho = np.min(rho)

    if max_value is not None:
        max_rho = np.max((np.log(max_value), min_rho))
    else:
        max_rho = np.max(rho)

    min_rho = np.min((min_rho, max_rho))

    fig, (ax, cax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [50, 1]}, figsize=(17, 4))

    im = NonUniformImage(ax,
                         interpolation='bilinear',
                         extent=(0, np.max(x), 0, np.min(z)),
                         cmap=cmap,
                         clim=(min_rho, max_rho)
                         )

    im.set_data(x, z, rho)
    ax.images.append(im)
    ax.set_xlim(0, np.max(x))
    ax.set_ylim(np.min(z), 0)

    cbar = plt.colorbar(im, cax=cax)

    cbar.set_ticks([0])

    iticks = np.linspace(min_rho, max_rho, 10)
    iticks_label = np.exp(np.linspace(min_rho, max_rho, len(iticks))).astype(np.int)
    cbar.set_ticks(iticks)
    cbar.set_ticklabels(iticks_label)

    if plot:
        plt.show()

    if return_as_pil:
        fig.canvas.draw()
        return Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    else:
        return fig


# img = show_zondres2d_model(r'F:\PycharmProjects\ER_dl\trash\model_9.dat', min_value=10, max_value=1000, plot=False, return_as_pil=True)
# img.show()


