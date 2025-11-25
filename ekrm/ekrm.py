import os

from . import utils

try:
    import cupy
except ModuleNotFoundError:
    print('Cupy not installed.')
    print('For CUDA 11.X, please use `pip install cupy-cuda11x`.')
    print('For CUDA 12.X, please use `pip install cupy-cuda12x`.')
    print('If you have multiple CUDA installed, please check the CUDA_PATH environment variable.')
    print(f'Current value of CUDA_PATH: {os.environ.get("CUDA_PATH", "")}')
    raise

import logging
import pickle
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, lru_cache
from itertools import combinations
from pathlib import Path
from typing import Iterable

import cupy as cp
import cupyx.scipy as sp
import cupyx.scipy.signal
import cv2
import matplotlib.pyplot as plt
import numba
import numpy as np
from PIL import Image
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from .constants import Channel, Colorspace, SaveAs
from .cupy.PIL import Image, ImageCms
from .cupy.matplotlib import pyplot as plt
from .cupy.sklearn.cluster import KMeans

try:
    import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger('ekrm')


def hex_color_to_rgb_color(hex_string: str) -> cp.array:
    return cp.array([int(hex_string[i: i + 2], 16) for i in range(1, len(hex_string), 2)]).astype(np.uint8)


class EKRM:
    def __init__(self, path: Path | str):
        path = path if isinstance(path, Path) else Path(path)
        self.raw_image_path: Path = path
        assert self.raw_image_path.exists() and self.raw_image_path.is_file(), f'File not exists: {path}'

    @dataclass
    class Group:
        name: str
        index: int
        color: str

        @cached_property
        def rgb_color(self) -> cp.ndarray:
            return hex_color_to_rgb_color(self.color).astype(cp.uint8)

        @cached_property
        def rgba_color(self) -> cp.ndarray:
            color = self.color
            if len(color.lstrip('#')) == 6:
                color = f'{color}FF'
            return hex_color_to_rgb_color(color)

    groups: Iterable[Group]
    color_space = Colorspace.lab
    index_fill_value = 0

    kmeans_n_cluster: int = 100

    _skip_saving = False
    _save_numpy = True
    _save_thumbnail = True
    _save_mpl_dpi = 1200

    @contextmanager
    def skip_saving_image(self):
        previous = self._skip_saving
        self._skip_saving = True
        yield
        self._skip_saving = previous

    def index_to_rgba(self, img: cp.ndarray):
        indices = [group.index for group in self.groups]
        colors = [group.rgba_color for group in self.groups]

        indices.append(self.index_fill_value)
        colors.append(hex_color_to_rgb_color('#FFFFFF00'))

        indices = cp.array(indices)
        colors = cp.array(colors)

        mapping = cp.zeros(int(max(indices)) + 1, dtype=cp.int_)

        unique_in_img = set(np.unique(img).tolist())

        expected_indices = set(indices.get().ravel().tolist())
        assert unique_in_img.issubset(expected_indices), (
            f"The unique values in the input image do not match the expected indices. "
            f"Unique values in image: {unique_in_img} "
            f"Expected indices: {expected_indices}. "
            f"Ensure that all unique values in the image correspond to defined group indices."
        )

        for i, g in enumerate(indices):
            mapping[g] = i
        group_index = mapping[img]
        img = cp.array(colors)[group_index]
        return img

    SupportedImage = Image.Image | cp.ndarray | plt.Figure | plt.Axes

    def save_image(self, img: SupportedImage, path: Path | str, *, image_type: SaveAs = None):
        if self._skip_saving:
            return
        path = Path(path)
        path = path if path.suffix else path.with_suffix('.png')
        path = self.get_opath(path, is_dir=False)
        if isinstance(img, plt.Axes):
            img: plt.Axes
            img = img.get_figure()
        if isinstance(img, plt.Figure):
            img: plt.Figure
            with plt.rc_context({'svg.fonttype': 'none'}):
                img.savefig(path.with_suffix('.svg'), dpi=self._save_mpl_dpi)
            img.savefig(path.with_suffix('.png'), dpi=self._save_mpl_dpi)
            return
        if not isinstance(img, Image.Image):
            if not isinstance(img, cp.ndarray):
                # noinspection PyTypeChecker
                img = cp.asarray(img)
            img: cp.ndarray
            channels = img.shape[-1]
            self._save_numpy and cp.savez_compressed(str(path.parent / f'{path.stem}.npz'), image=img)
            if image_type is None:
                if img.dtype == cp.bool_:
                    image_type = SaveAs.bit
                elif cp.min(img) >= 0 and cp.max(img) <= 1:
                    image_type = SaveAs.gray
                elif cp.all(cp.in1d(img, cp.array([g.index for g in self.groups] + [cp.nan, 0]))):
                    image_type = SaveAs.index
            match image_type:
                case SaveAs.bit | SaveAs.gray:
                    img = img * 255
                case SaveAs.index:
                    img = self.index_to_rgba(img)
            img: Image.Image = Image.fromarray(img)
        img.save(str(path))
        if self._save_thumbnail and any(((width := img.width) > 500, (height := img.height) > 500)):
            scale = cp.max(cp.ceil(cp.array((width, height)) / 500))
            width, height = cp.round(width / scale), cp.round(height / scale)
            corner_dir = path.parent / 'left-top-corners'
            corner_dir.mkdir(parents=True, exist_ok=True)
            img.crop((0, 0, 500, 500)).save(corner_dir / path.name)
            thumbnail = img.copy()
            thumbnail.thumbnail((width, height))
            thumbnail_dir = path.parent / 'thumbnails'
            thumbnail_dir.mkdir(parents=True, exist_ok=True)
            thumbnail.save(thumbnail_dir / path.name)

    @lru_cache(maxsize=20)
    def get_image_array(self, specify: Colorspace | Channel) -> cp.ndarray:
        match specify:
            case Colorspace.rgb:
                with Image.open(self.raw_image_path, 'r') as img:
                    return cp.asarray(img)
            case Colorspace.hsv:
                img = self.get_image_array(specify.rgb)
                return cp.asarray(Image.fromarray(img.get()).convert('HSV'))
            case Colorspace.grey:
                img = self.get_image_array(specify.rgb)
                return cp.asarray(Image.fromarray(img.get()).convert('L')).reshape((*img.shape[:-1], 1))
            case Colorspace.lab:
                img = self.get_image_array(specify.rgb)
                img = Image.fromarray(img.get())
                in_profile = ImageCms.createProfile("sRGB")
                out_profile = ImageCms.createProfile("LAB")
                transform = ImageCms.buildTransformFromOpenProfiles(in_profile, out_profile, "RGB", "LAB")
                img = ImageCms.applyTransform(img, transform)
                img = cp.asarray(img)
                return img
            case Channel.rgb_red:
                return self.get_image_array(Colorspace.rgb)[:, :, 0]
            case Channel.rgb_green:
                return self.get_image_array(Colorspace.rgb)[:, :, 1]
            case Channel.rgb_blue:
                return self.get_image_array(Colorspace.rgb)[:, :, 2]
            case Channel.greyscale_grey:
                return self.get_image_array(Colorspace.grey)[:, :, 0]
            case Channel.hsv_hue:
                return self.get_image_array(Colorspace.hsv)[:, :, 0]
            case Channel.hsv_saturation:
                return self.get_image_array(Colorspace.hsv)[:, :, 1]
            case Channel.hsv_value:
                return self.get_image_array(Colorspace.hsv)[:, :, 2]
            case Channel.lab_lightness_star:
                return self.get_image_array(Colorspace.lab)[:, :, 0]
            case Channel.lab_a_star:
                return self.get_image_array(Colorspace.lab)[:, :, 1]
            case Channel.lab_b_star:
                return self.get_image_array(Colorspace.lab)[:, :, 2]
            case _:
                raise NotImplementedError(f'Unknown image type: {specify}')

    @cached_property
    def pixel_count(self):
        return int(np.prod(self.get_image_array(Colorspace.rgb).shape[:2]))

    def get_ipath(self, path: str | Path = None) -> Path:
        result = self.raw_image_path.parent / 'output' / self.raw_image_path.name
        result = result if path is None else result / path
        return result

    def get_output_root(self) -> Path:
        return self.raw_image_path.parent / 'output' / self.raw_image_path.name

    _pytest_skip_mk_dir = False

    def get_opath(self, path: str | Path = None, *, is_dir=True) -> Path:
        root = self.get_output_root()
        path = root if path is None else root / path
        self._pytest_skip_mk_dir or (path if is_dir else path.parent).mkdir(parents=True, exist_ok=True)
        return path

    def describe_raw_image(self, dpi=100, colorspace: Colorspace = None):
        if colorspace is None:
            [self.describe_raw_image(dpi, colorspace) for colorspace in Colorspace]
            return
        out = self.get_opath('describe-raw-image')
        colorspace_out = self.get_opath(out / f'{colorspace.label}')
        for channel in colorspace.channels:
            bins = list(range(0, 256))
            img = self.get_image_array(channel)
            img_to_save = img.copy()
            match channel:
                case Channel.lab_a_star | Channel.lab_b_star:
                    bins = list(range(0, 101))
                    img = (img - 128) % 100
                    img_to_save = cp.round(img * 2.55)
            self.save_image(img_to_save, colorspace_out / f'channel-{channel.value}')
            plt.figure()
            plt.hist(img.reshape(-1).get(), rwidth=1, bins=bins)
            plt.xlabel(f'{channel.value}')
            plt.ylabel(f'Count of pixel')
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            self.save_image(plt.gca(), colorspace_out / f'channel-{channel.value}-hist')
            plt.close()
        selected_channels: tuple[Channel, ...]
        for selected_channels in (*combinations(colorspace.channels, 2), *combinations(colorspace.channels, 3)):
            arrays = [*Colorspace.rgb.channels, *selected_channels]
            arrays = [self.get_image_array(a).reshape(-1) for a in arrays]
            arrays = cp.column_stack(arrays)
            arrays = np.unique(arrays.get(), axis=0)
            r, g, b, *arrays = [arrays[:, i] for i in range(arrays.shape[1])]
            arrays = list(arrays)
            for index, (channel, array) in enumerate(zip(selected_channels, arrays)):
                match channel:
                    case channel.lab_a_star | channel.lab_b_star:
                        arrays[index] = (array + 128 + 256) % 256
            figure: plt.Figure = plt.figure(dpi=dpi)
            figure.suptitle(f'Colorspace {colorspace.label}')
            s, c = np.zeros_like(r) + 0.3, np.array((r, g, b)).T / 255
            channel_names = tuple(c.value for c in selected_channels)
            match len(selected_channels):
                case 2:
                    x, y = arrays
                    ax: Axes = figure.add_subplot()
                    # noinspection PyTypeChecker
                    ax.set_aspect('equal'), [f(0, 255) for f in (ax.set_xlim, ax.set_ylim)]
                    ax.scatter(x, y, s=s, c=c)
                    [f(v) for f, v in zip((ax.set_xlabel, ax.set_ylabel), channel_names)]
                case 3:
                    match colorspace:
                        case colorspace.hsv:
                            h, s, v = arrays
                            theta, rho = h / 255 * np.pi * 2, s * (v / 255) / 2
                            x, y, z = rho * np.cos(theta), rho * np.sin(theta), v
                            x, y = x + 128, y + 128
                            arrays = x, y, z
                            channel_names = ('Hue (radian)', 'Saturation (radius)', 'Value')
                        case colorspace.lab:
                            l, a, b = arrays
                            arrays = a, b, l
                            channel_names = ('A*', 'B*', 'Lightness*')
                    ax: Axes3D = figure.add_subplot(projection='3d')
                    [f(0, 255) for f in (ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d)]
                    ax.scatter(*arrays, s=s, c=c)
                    [f(v) for f, v in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel), channel_names)]
                case _:
                    raise RuntimeError('Never')
            self.save_image(ax, colorspace_out / f'distribution-{"-".join([c.value for c in selected_channels])}')
            plt.close(figure)

    def get_kmeans_dir(self, *, is_input=False) -> Path:
        dir_name = f'{self.color_space.label}-cluster-{self.kmeans_n_cluster}'
        return self.get_ipath(dir_name) if is_input else self.get_opath(dir_name)

    @classmethod
    def color_to_int(cls, img: cp.ndarray) -> cp.ndarray:
        return utils.color_to_int(img)

    @classmethod
    def int_to_color(cls, colors_int: cp.ndarray, channel_count: int = 3) -> cp.ndarray:
        return utils.int_to_color(colors_int, channel_count)

    def get_unique_colors(self, img: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        return utils.get_unique_colors(img)

    def kmeans_fit(self, use_cache=True):
        if (in_path := self.get_kmeans_dir(is_input=True) / 'model.pkl').exists() and use_cache:
            with open(in_path, 'rb') as f:
                return pickle.load(f)

        model = KMeans(n_clusters=self.kmeans_n_cluster, random_state=8, max_iter=1000, tol=1E-10)
        img = self.get_image_array(self.color_space)
        colors, count = self.get_unique_colors(img)
        logger.info(f'Learning')
        model.fit(colors, sample_weight=count)
        with open(self.get_kmeans_dir() / 'model.pkl', 'wb') as f:
            pickle.dump(model, f)
        return model

    def predict(self, use_cache: bool = True, image: cp.ndarray = None) -> cp.ndarray:
        npy_input_path = self.get_kmeans_dir(is_input=True) / 'predicted.npz'
        npy_output_path = self.get_kmeans_dir(is_input=False) / 'predicted'
        if use_cache and npy_input_path.exists():
            logger.info('Loading cached prediction result ...')
            return self.load_npz(npy_input_path)
        logger.info('Predicting ...')
        arr = self.get_image_array(self.color_space) if image is None else image
        channel_count = arr.shape[-1]
        to_predict: cp.ndarray = arr.reshape((-1, channel_count))
        predicted = self.kmeans_fit().predict(to_predict)
        predicted = predicted.reshape(arr.shape[:-1])
        self.save_image(predicted, npy_output_path)
        return predicted

    @classmethod
    def color_data_to_square_image(cls, color_data: cp.ndarray) -> cp.ndarray:
        channel_count = color_data.shape[-1]
        indexes = cp.argsort(color_data, axis=0)
        color_data = cp.take_along_axis(color_data, indexes, axis=0)
        pixel_count = color_data.shape[0]
        side_length = cp.ceil(cp.sqrt(pixel_count)).astype(cp.int_)
        shape = (int(side_length ** 2 - pixel_count), channel_count)
        color_data = cp.vstack((color_data, cp.ones(shape) * 255))
        color_data = color_data.reshape((int(side_length), int(side_length), channel_count)).astype(cp.uint8)
        return color_data

    def generate_predict_results(self):
        out = self.get_kmeans_dir() / datetime.now().strftime('%y%m%d-%H%M%S')
        color_dir = self.get_opath(out / 'colors')
        logger.info('Generating predict results as images ...')
        predicted = self.predict()
        rgb = self.get_image_array(Colorspace.rgb)
        states = self._save_numpy, self._save_thumbnail
        self._save_numpy, self._save_thumbnail = False, False
        logger.info('Saving images of each cluster ...')
        bar = tqdm.tqdm(total=self.kmeans_n_cluster).__enter__() if (use_tqdm := tqdm is not None) else None
        for i in range(self.kmeans_n_cluster):
            selector = predicted == i
            color_data = rgb[selector]
            pixel_count = color_data.shape[0]
            color_data = self.color_data_to_square_image(color_data)
            name = f'{pixel_count / self.pixel_count * 10000:04.0f}-color{i}'
            self.save_image(color_data, color_dir / name)
            use_tqdm and bar.update()
        use_tqdm and bar.__exit__(None, None, None)
        [self.get_opath(out / f'{g.index}-{g.name}') for g in self.groups]
        self._save_numpy, self._save_thumbnail = states

    def kmeans_get_classification_dir(self, *, is_input=False) -> Path:
        dirs = [path for path in self.get_kmeans_dir(is_input=is_input).glob('*')
                if path.is_dir() and re.match(r'^\d{6}-\d{6}$', path.name)]
        if is_input:
            assert dirs, 'Should generate and classify the images first'
        dirs.sort(key=lambda i: i.stat().st_ctime)
        return dirs[-1] if dirs else None

    def load_npz(self, path: Path | str, use_numpy=False) -> cp.ndarray:
        path = self.get_ipath(path) if isinstance(path, str) else path
        assert path.exists(), f'File not found: {path}'
        # noinspection PyProtectedMember
        array: cp._numpy.lib.npyio.NpzFile = cp.load(path).npz_file
        array: cp.ndarray = array[array.files[0]]
        return array if use_numpy else cp.array(array)

    def get_mapping_dict(self):
        in_dir = self.kmeans_get_classification_dir(is_input=True)
        files = in_dir.glob(r'[0-9]*-*/[0-9]*-color[0-9]*.png')
        files = (f'{f.parent.name}-{f.stem}' for f in files)
        regex = re.compile(r'^(\d{1,2})-.*?-\d{4}-color(\d+)$')
        files = (regex.match(f) for f in files)
        files = (f.groups() for f in files)
        files = (map(int, f) for f in files)
        mapping_dict = {color_index: group_index for group_index, color_index in files}
        return mapping_dict

    def kmeans_evaluate(self, use_cache=True, image: cp.ndarray = None) -> cp.ndarray:
        logger.info('Getting K-Means result after manually classification  ...')
        in_dir, opath = self.kmeans_get_classification_dir(is_input=True), self.kmeans_get_classification_dir()
        groups = self.groups
        result_name = 'classification-result'
        if use_cache and (result_path := in_dir / f'{result_name}.npz').exists():
            return self.load_npz(result_path)
        mapping_dict = self.get_mapping_dict()
        unmatched = list(groups)[0].index
        mapping = cp.array(tuple(mapping_dict.get(i, unmatched) for i in range(self.kmeans_n_cluster)), dtype=cp.int_)
        predicted = self.predict(use_cache=use_cache, image=image)

        if image is None:
            combined_colors_dir = self.get_opath(opath / 'combined-colors')
            rgb = self.get_image_array(Colorspace.rgb)
            combined_dict = {g.index: [] for g in groups}
            for i in range(self.kmeans_n_cluster):
                combined_dict[mapping_dict[i]].append(rgb[predicted == i])
            for g in groups:
                combined = combined_dict[g.index]
                if not combined:
                    continue
                combined = np.vstack(combined)
                combined = self.color_data_to_square_image(combined)
                self.save_image(combined, combined_colors_dir / f'{g.name}')

        predicted = mapping[predicted]
        self.save_image(self.get_image_array(Colorspace.rgb), opath / 'source')
        self.save_image(predicted, opath / result_name)

        return predicted

    gaussian_convolution_radius = 5

    def get_gaussian_convolution_kernel(self, kernel_radius: int = None) -> cp.ndarray:
        if kernel_radius > 100:
            raise ValueError('Does not support such a large kernel')
        kernel_radius = self.gaussian_convolution_radius if kernel_radius is None else kernel_radius
        kernel_shape = kernel_radius * 2
        kernel_sigma = kernel_radius / 3
        # noinspection PyTypeChecker
        kernel_x, kernel_y = cp.meshgrid(cp.arange(kernel_shape), cp.arange(kernel_shape))
        kernel_distance = cp.sqrt((kernel_x - kernel_radius) ** 2 + (kernel_y - kernel_radius) ** 2)
        kernel = 1 / cp.sqrt(2 * cp.pi) / kernel_radius * cp.exp(-kernel_distance ** 2 / 2 / kernel_sigma ** 2)
        kernel = kernel / cp.sum(kernel)
        kernel = kernel * 1000000000
        kernel = cp.round_(kernel).astype(cp.int_)
        return kernel

    @cached_property
    def gaussian_convolution_kernel(self) -> cp.ndarray:
        return self.get_gaussian_convolution_kernel()

    @cached_property
    def convolution_result(self) -> cp.ndarray:
        return self.convolve()

    def convolve(self, src: cp.ndarray = None, kernel_radius: int = None) -> cp.ndarray:
        radius = kernel_radius
        radius = self.gaussian_convolution_radius if radius is None else radius
        logger.info(f'Getting convolution with radius {radius} ...')
        src = self.kmeans_evaluate() if src is None else src
        out_dir = self.kmeans_get_classification_dir() / f'convolution/kernel-radius-{radius}'
        time_str = datetime.now().strftime('%y%m%d-%H%M%S')
        # noinspection PyUnresolvedReferences
        colormap = (cp.round_(cp.array(plt.cm.viridis.colors) * 255)).astype(cp.uint8)
        kernel = self.get_gaussian_convolution_kernel(radius)
        max_index = cp.zeros_like(src, dtype=cp.uint8)
        max_value = cp.zeros_like(src, dtype=cp.float64)
        radius_m1 = radius - 1
        for group in self.groups:
            layer = cp.equal(src, group.index).astype(cp.float64)
            layer = sp.signal.convolve2d(layer, kernel, 'full')[radius_m1:-radius_m1, radius_m1:-radius_m1]
            layer = layer[:-1, :-1] + layer[1:, :-1] + layer[:-1, 1:] + layer[1:, 1:]
            selected = layer >= max_value
            max_index[selected] = group.index
            max_value[selected] = layer[selected]
            self.save_image(colormap[cp.round_(layer * 255).astype(cp.uint8)], out_dir / f'{time_str}-{group.name}.png')
        self.save_image(max_index, out_dir / f'{time_str}.png')
        del src, kernel, layer, max_value, selected
        self.clean_gpu_memory()
        return max_index

    @classmethod
    def clean_gpu_memory(cls):
        cp.get_default_memory_pool().free_all_blocks()

    @classmethod
    def np_indexed_image_to_contours(cls, image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        if isinstance(image, cp.ndarray):
            image: cp.ndarray
            image = image.get()
        assert isinstance(image, np.ndarray), type(image)
        all_contours = []
        all_meta = np.zeros((0, 6), dtype=np.int_)
        for color_index in np.unique(image):
            contours, layer_meta = cls.np_bit_image_to_contours(image == color_index, mode=cv2.RETR_CCOMP)
            layer_meta = layer_meta.reshape((-1, 4))

            parsed_contours, parsed_meta = [], []
            for contour, contour_meta in zip(contours, layer_meta):
                [(parsed_contours.append(c), parsed_meta.append(contour_meta)) for c in cls.__np_split_contour(contour)]
            contours, layer_meta = [np.array(c) for c in parsed_contours], np.array(parsed_meta)

            mapping = np.arange((row := all_meta.shape[0]), row + layer_meta.shape[0])
            layer_meta = np.where(layer_meta == -1, -1, mapping[layer_meta])

            is_outer = layer_meta[:, -1] == -1
            layer_meta = layer_meta[is_outer, :]
            contours = [c for i, c in zip(is_outer, contours) if i]

            area = np.array([cls.np_get_area_of_contour(c) for c in contours]).reshape((-1, 1))
            layer_meta = np.hstack((layer_meta, np.zeros_like(area) + color_index, area))

            all_contours.extend(contours)
            all_meta = np.vstack((all_meta, layer_meta))
        indexes = np.argsort(all_meta[:, -1])[::-1]
        all_meta = all_meta[indexes, :]
        all_meta = all_meta.reshape((1, -1, 6))
        all_contours = [all_contours[i] for i in indexes]
        return all_contours, all_meta.reshape((1, -1, 6))

    @classmethod
    def np_bit_image_to_contours(cls, image: np.ndarray, mode=cv2.RETR_TREE):
        image = image.astype(np.uint8)
        assert isinstance(image, np.ndarray), type(image)
        assert np.alltrue(np.isin((unique := np.unique(image)), np.array([0, 1]))), unique
        contours, hierarchy = cv2.findContours(image, mode, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy

    @classmethod
    def np_get_area_of_contour(cls, contour: np.ndarray) -> int:
        contour: np.ndarray = contour.copy().reshape((-1, 2))
        contour -= np.min(contour, 0)
        shape = tuple(int(i) for i in (np.max(contour, 0) + 1))[::-1]
        arr = np.zeros(shape, dtype=np.int_)
        arr = cv2.fillPoly(arr, [contour], 1)
        return int(np.sum(arr))

    @classmethod
    def __np_split_contour(cls, contour):
        contour = [(x, y) for x, y in contour[:, 0, :]]
        if len(set(contour)) == len(contour):
            return [contour]
        else:
            repeated = [point for point, count in Counter(contour).items() if count == 2]
            split = [contour]
            for repeat in repeated:
                temp = []
                for c in split:
                    if c.count(repeat) != 2:
                        temp.append(c)
                        continue
                    start = c.index(repeat)
                    end = c.index(repeat, start + 1)
                    temp.append([*c[0:start], *c[end:]])
                    temp.append(c[start:end])
                split = temp
        return split

    @classmethod
    def np_contours_to_indexed_image(cls, contours: list[np.ndarray], hierarchy: np.array, shape, fill_value=0):
        result_layer = np.full(shape, fill_value=fill_value, dtype=np.int_)
        for contour, color_index in zip(contours, hierarchy[0, :, -2]):
            if contour.shape[0] == 0:
                continue
            cv2.fillPoly(result_layer, [contour], int(color_index))
        return np.array(result_layer)

    @staticmethod
    @numba.njit()
    def _numba_np_indexed_to_boundary(output: np.ndarray):
        pairs = ((
                     [[0, 1, 1, 0],
                      [0, 1, 1, 0]],
                     [[0, 1, 0, 0],
                      [0, 1, 0, 0]]
                 ), (
                     [[2, 1, 2],
                      [0, 1, 0],
                      [0, 0, 0]],
                     [[2, 1, 2],
                      [0, 0, 0],
                      [0, 0, 0]]
                 ), (
                     [[0, 2, 2],
                      [2, 1, 1],
                      [2, 1, 2]],
                     [[0, 2, 2],
                      [2, 0, 1],
                      [2, 1, 2]],
                 ))
        pairs = [(np.array(a), np.array(b)) for a, b in pairs]
        for previous_tpl, replaced_tpl in pairs:
            width, height = previous_tpl.shape
            selector: np.ndarray = previous_tpl != 2
            count = width * height - np.sum(previous_tpl == 2)
            for i in range(4):
                for x in range(output.shape[0] - width + 1):
                    for y in range(output.shape[1] - height + 1):
                        sub_arr: np.ndarray = output[x:x + width, y:y + height]
                        if np.sum(sub_arr == previous_tpl) == count:
                            sub_arr = np.where(selector, replaced_tpl, sub_arr)
                            output[x:x + width, y:y + height] = sub_arr
                output = np.rot90(output)
        return output

    def np_indexed_to_boundary(self, img: np.ndarray):
        img = utils.ensure_numpy(img)
        output = np.zeros_like(img)
        contours, meta = self.np_indexed_image_to_contours(img)
        cv2.polylines(output, contours, True, 1)
        output = self._numba_np_indexed_to_boundary(output)
        return output

    def _fix_noizy_convolve(self, image: cp.ndarray, radius: int, out: Path) -> cp.ndarray:
        self.save_image(image, out.parent / f'{out.stem}-1-before-convolve')
        with self.skip_saving_image():
            image = self.convolve(image, radius)
        self.save_image(image, out.parent / f'{out.stem}-2-after-convolve')
        return image

    def _fix_noizy_domain_area(self, image: cp.ndarray, pixel_count: int, out: Path, max_iter: int):
        fill_value = -99
        for index in range(max_iter):
            logger.info(f'Fixing noizy pixels: batch {index}')
            contours, meta = self.np_indexed_image_to_contours(image.get())
            v = np.where(meta[0, :, -1] < pixel_count)[0]
            if v.shape[0] == 0:
                continue
            count = np.min(v)
            new_image = cp.array(self.np_contours_to_indexed_image(contours[:count], meta[:, :count, :], image.shape,
                                                                   fill_value=fill_value))
            assert new_image.shape == image.shape, new_image.shape
            convolved = self.convolve(new_image)
            unique_values = {i for i in cp.unique(cp.reshape(convolved, -1)).get()}
            assert unique_values.issubset({g.index for g in self.groups}), unique_values
            assert convolved.shape == image.shape
            new_image = cp.where(cp.abs(new_image - fill_value) < 1E-3, convolved, new_image)
            unique_values = {i for i in cp.unique(cp.reshape(new_image, -1)).get()}
            assert unique_values.issubset({g.index for g in self.groups}), unique_values
            self.save_image(image, out.parent / f'{out.stem}-iter-{index}')
            if cp.array_equal(image, new_image):
                break
            image = new_image
        else:
            pass
            logger.warning('Fix noize pixels warning: max recursion reached')
        return image

    fix_noize_convolve_radius = 5

    def fix_noizy_pixels(self, *,
                         convolve_radius: int = None,
                         image: cp.ndarray = None,
                         pixel_count: int = None,
                         max_iter: int = None):
        convolve_radius = self.fix_noize_convolve_radius if convolve_radius is None else convolve_radius
        image = self.kmeans_evaluate() if image is None else image
        pixel_count = int(np.ceil(image.size / 10000)) if pixel_count is None else pixel_count
        output_dir_name = f'fix-noize-initial-convolve-{convolve_radius}-pixel-{pixel_count}'
        out = self.kmeans_get_classification_dir() / output_dir_name
        max_iter = int(np.ceil(np.sqrt(pixel_count))) if max_iter is None else max_iter
        image = self._fix_noizy_convolve(image, convolve_radius, self.get_opath(out / 'stage-1'))
        image = self._fix_noizy_domain_area(image, pixel_count, self.get_opath(out / 'stage-2'), max_iter)
        image = self._fix_noizy_convolve(image, convolve_radius, self.get_opath(out / 'stage-3'))
        image = self._fix_noizy_domain_area(image, pixel_count, self.get_opath(out / 'stage-4'), max_iter)
        self.save_image(image, out / 'fixed')
        self.clean_gpu_memory()
        return image

    def batch_process(
            self,
            src_dir: Path, dst_dir: Path,
            convolve_radius: int = None, pixel_count: int = None, max_iter: int = None,
    ):
        # 实现批量处理：遍历src_dir中的所有png和jpg文件
        src_dir, dst_dir = Path(src_dir), Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        image_files: list[Path] = list(src_dir.glob('*.png')) + list(src_dir.glob('*.jpg'))

        for src_file in image_files:
            dst_file = dst_dir / src_file.relative_to(src_dir).with_suffix('.png')
            obj = EKRM(src_file)
            with self.skip_saving_image():
                image = obj.get_image_array(self.color_space)
                image = self.kmeans_evaluate(use_cache=False, image=image)
                image = self.fix_noizy_pixels(
                    image=image,
                    convolve_radius=convolve_radius,
                    pixel_count=pixel_count,
                    max_iter=max_iter
                )
            self.save_image(image, dst_file)
