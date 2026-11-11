"""
Read and write rectification matrices (rectmats) as FITS.

A rectmat file is what the caldb serves for cal_type rectmat_opt / rectmat_c2.
Everything needed to use the matrix is in the file, so the DRP does not need
per-mode config for cube shapes or wavelength ranges:

    PRIMARY     header only: identity, cube layout and how the matrix was built
    MATRIX_COO  the sparse matrix (see sparse_matrix_utils.pack_matrix_fits)
    WAVELENGTH  one row per cube slice: CHANNEL, WAVE [um] (+ MONOWAVE for
                rectmats built directly from the monochromator frames)
    LENSMAP     (ny, nx) number of cube slices each lenslet has a response in;
                0 means the lenslet is not in the matrix at all
    SOURCES     optional: the monochromator frames the matrix was built from

Flux index convention (both kinds): flux = x + y*nx + lam*nx*ny.
OPT rectmats are (n_flux, n_det) and C2 rectmats are (n_det, n_flux), with
detector index = x + y*NDETX.

Legacy .npz rectmats (scipy save_npz, optionally with a 'cube_shape' key from
the old save_rectmat) can still be read; they have no wavelength table.
"""
import os
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from astropy.time import Time
from scipy import sparse

from scalesdrp.core.sparse_matrix_utils import pack_matrix_fits, unpack_matrix_fits

N_DET_Y = 2048
N_DET_X = 2048
N_DET_PIX = N_DET_Y * N_DET_X

WAVE_EXTNAME = "WAVELENGTH"
LENSMAP_EXTNAME = "LENSMAP"
SOURCES_EXTNAME = "SOURCES"

# kind -> (caldb cal_type, MORIENT)
KINDS = {
    "OPT": ("rectmat_opt", "FLUX,DET"),
    "C2": ("rectmat_c2", "DET,FLUX"),
}

# Relative tolerance on the step size for a wavelength grid to count as linear
UNIFORM_RTOL = 1e-6

FORMAT_VERSION = 1


@dataclass
class Rectmat:
    """A loaded rectmat and what the file says about it."""
    matrix: sparse.spmatrix
    cube_shape: tuple
    kind: str
    wavelengths: np.ndarray = None
    lensmap: np.ndarray = None
    header: fits.Header = field(default_factory=fits.Header)
    path: str = None

    @property
    def nlam(self):
        return self.cube_shape[0]

    @property
    def lenslet_grid(self):
        return self.cube_shape[1:]

    def to_opt(self):
        """The matrix in OPT orientation (n_flux, n_det)."""
        return self.matrix if self.kind == "OPT" else self.matrix.T.tocsr()


def merge_header(hdr, extra):
    """Add keywords from a Header (cards, incl. HISTORY) or a dict to hdr."""
    if extra is None:
        return hdr
    if isinstance(extra, fits.Header):
        for card in extra.cards:
            if card.keyword in ("COMMENT", "HISTORY", ""):
                hdr.append(card)
            else:
                hdr[card.keyword] = (card.value, card.comment)
    else:
        for key, value in extra.items():
            hdr[key] = value
    return hdr


def flux_axis(matrix, n_det_pix=N_DET_PIX):
    """Which axis of a rectmat is the flux axis (0 for OPT, 1 for C2)."""
    if matrix.shape[1] == n_det_pix and matrix.shape[0] != n_det_pix:
        return 0
    if matrix.shape[0] == n_det_pix and matrix.shape[1] != n_det_pix:
        return 1
    raise ValueError(f"Cannot tell which rectmat axis {matrix.shape} is the "
                     f"detector ({n_det_pix} pixels).")


def kind_from_shape(matrix, n_det_pix=N_DET_PIX):
    return "OPT" if flux_axis(matrix, n_det_pix) == 0 else "C2"


def is_uniform(wavelengths, rtol=UNIFORM_RTOL):
    """True if the wavelength grid is linear (constant step) to within rtol."""
    w = np.asarray(wavelengths, dtype=float)
    if w.size < 3:
        return True
    step = np.diff(w)
    mean = step.mean()
    return bool(mean != 0 and np.all(np.abs(step - mean) <= rtol * abs(mean)))


def lensmap_from_matrix(matrix, cube_shape):
    """
    (ny, nx) count of cube slices in which each lenslet has a nonzero response.
    """
    nlam, ny, nx = cube_shape
    axis = flux_axis(matrix)
    coo = matrix.tocoo()
    flux = coo.row if axis == 0 else coo.col
    flux = np.unique(flux[coo.data != 0])
    counts = np.bincount(flux % (ny * nx), minlength=ny * nx)
    return counts.reshape(ny, nx).astype(np.int16)


def wavelength_hdu(wavelengths, monowave=None):
    w = np.asarray(wavelengths, dtype=np.float64)
    cols = [
        fits.Column(name="CHANNEL", format="J", array=np.arange(w.size, dtype=np.int32)),
        fits.Column(name="WAVE", format="D", unit="um", array=w),
    ]
    if monowave is not None:
        mw = np.asarray(monowave, dtype=np.float64)
        if mw.shape != w.shape:
            raise ValueError(f"monowave has {mw.size} entries, wavelengths has {w.size}.")
        cols.append(fits.Column(name="MONOWAVE", format="D", unit="um", array=mw))
    hdu = fits.BinTableHDU.from_columns(cols, name=WAVE_EXTNAME)
    hdr = hdu.header
    hdr["NLAM"] = (w.size, "number of cube slices")
    hdr["WAVEUNIT"] = ("um", "wavelength unit")
    hdr["UNIFORM"] = (is_uniform(w), "wavelength grid has a constant step")
    hdr["WAVSTART"] = (float(w[0]), "wavelength of the first slice [um]")
    hdr["WAVEND"] = (float(w[-1]), "wavelength of the last slice [um]")
    if w.size > 1:
        hdr["DWAVE"] = (float((w[-1] - w[0]) / (w.size - 1)), "mean step [um/slice]")
    hdr.add_comment("WAVE is the wavelength of cube slice CHANNEL (0 indexed).")
    return hdu


def _sources_hdu(sources):
    """sources: list of dicts with filename / monowave [um] / date_obs."""
    names = [str(s.get("filename", "")) for s in sources]
    width = max([len(n) for n in names] + [1])
    cols = [
        fits.Column(name="FILENAME", format=f"{width}A", array=np.array(names)),
        fits.Column(name="MONOWAVE", format="D", unit="um",
                    array=np.array([float(s.get("monowave", np.nan)) for s in sources])),
        fits.Column(name="DATE-OBS", format="30A",
                    array=np.array([str(s.get("date_obs", "")) for s in sources])),
    ]
    return fits.BinTableHDU.from_columns(cols, name=SOURCES_EXTNAME)


def write_rectmat(path, matrix, cube_shape, wavelengths, kind=None, header=None,
                  monowave=None, sources=None, overwrite=True):
    """
    Write a rectmat FITS file.

    Parameters
    ----------
    path : str
        Output file (.fits).
    matrix : scipy.sparse matrix
        OPT (n_flux, n_det) or C2 (n_det, n_flux). Stored entries that are
        exactly zero are dropped; they carry no information but can make up
        most of a matrix.
    cube_shape : (nlam, ny, nx)
        How the flux axis maps onto the cube.
    wavelengths : array of nlam floats [um]
        Wavelength of each cube slice.
    kind : 'OPT' or 'C2', optional
        Checked against the matrix orientation; inferred if not given.
    header : astropy.io.fits.Header or dict, optional
        Extra primary-header keywords: instrument state (STATEID, IFSMODE,
        MODSLNAM, ...), dates, and the parameters used to build the matrix.
    monowave : array of nlam floats [um], optional
        Monochromator wavelength of the frame each slice came from (for
        rectmats built directly from the frames, not interpolated).
    sources : list of dict, optional
        Frames the matrix was built from (filename, monowave, date_obs).

    Returns
    -------
    str : the path written.
    """
    cube_shape = tuple(int(n) for n in cube_shape)
    if len(cube_shape) != 3:
        raise ValueError(f"cube_shape must be (nlam, ny, nx), got {cube_shape}.")
    shape_kind = kind_from_shape(matrix)
    kind = kind or shape_kind
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}, got {kind!r}.")
    if kind != shape_kind:
        raise ValueError(f"A {kind} rectmat must be {KINDS[kind][1]}, but this "
                         f"matrix has shape {matrix.shape}.")
    n_flux = matrix.shape[flux_axis(matrix)]
    if n_flux != int(np.prod(cube_shape)):
        raise ValueError(f"Rectmat has {n_flux} flux elements but cube_shape "
                         f"{cube_shape} implies {int(np.prod(cube_shape))}.")
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if wavelengths.shape != (cube_shape[0],):
        raise ValueError(f"Expected {cube_shape[0]} wavelengths, got {wavelengths.shape}.")

    m = matrix.tocsr(copy=True)
    n_stored = m.nnz
    m.eliminate_zeros()

    hdr = merge_header(fits.Header(), header)
    cal_type, orient = KINDS[kind]
    hdr["INSTRUME"] = (hdr.get("INSTRUME", "SCALES"), "instrument")
    hdr["CALTYPE"] = (cal_type, "caldb cal_type")
    hdr["RMATKIND"] = (kind, "OPT (optimal extract.) or C2 (chi2 model)")
    hdr["RMATFMT"] = (FORMAT_VERSION, "rectmat file format version")
    hdr["MORIENT"] = (orient, "matrix axes (rows,columns)")
    hdr["NLAM"] = (cube_shape[0], "cube slices (wavelengths)")
    hdr["NLENSY"] = (cube_shape[1], "lenslet grid rows (cube y)")
    hdr["NLENSX"] = (cube_shape[2], "lenslet grid columns (cube x)")
    hdr["NDETY"] = (N_DET_Y, "detector rows")
    hdr["NDETX"] = (N_DET_X, "detector columns")
    hdr["NSTORED"] = (n_stored, "entries stored before dropping explicit zeros")
    hdr["DATE"] = (Time.now().isot, "file creation time (UTC)")
    hdr.add_comment("flux index = x + y*NLENSX + lam*NLENSX*NLENSY")
    hdr.add_comment("detector index = x + y*NDETX")
    lensmap = lensmap_from_matrix(m, cube_shape)
    hdr["NLENSUSE"] = (int(np.count_nonzero(lensmap)), "lenslets with any response")

    hdul = pack_matrix_fits(m, header=hdr)
    hdul.append(wavelength_hdu(wavelengths, monowave))
    lens_hdu = fits.ImageHDU(lensmap, name=LENSMAP_EXTNAME)
    lens_hdu.header["EXTDESC"] = "cube slices with a response, per lenslet"
    hdul.append(lens_hdu)
    if sources:
        hdul.append(_sources_hdu(sources))

    tmp = f"{path}.tmp{os.getpid()}"
    hdul.writeto(tmp, overwrite=True)
    if os.path.exists(path) and not overwrite:
        os.remove(tmp)
        raise FileExistsError(path)
    os.replace(tmp, path)
    return path


def _read_fits(path):
    with fits.open(path, memmap=False) as hdul:
        matrix = unpack_matrix_fits(hdul)
        hdr = hdul[0].header.copy()
        cube_shape = (hdr["NLAM"], hdr["NLENSY"], hdr["NLENSX"])
        wavelengths = None
        if WAVE_EXTNAME in hdul:
            wavelengths = np.asarray(hdul[WAVE_EXTNAME].data["WAVE"], dtype=np.float64)
        lensmap = np.asarray(hdul[LENSMAP_EXTNAME].data) if LENSMAP_EXTNAME in hdul else None
    kind = hdr.get("RMATKIND") or kind_from_shape(matrix)
    if matrix.shape[flux_axis(matrix)] != int(np.prod(cube_shape)):
        raise ValueError(f"{os.path.basename(path)}: header cube shape {cube_shape} "
                         f"does not match the matrix {matrix.shape}.")
    return Rectmat(matrix, cube_shape, kind, wavelengths, lensmap, hdr, path)


def _read_npz(path, lenslet_grid=None):
    matrix = sparse.load_npz(path)
    n_flux = matrix.shape[flux_axis(matrix)]
    with np.load(path) as z:
        stored = tuple(int(n) for n in z["cube_shape"]) if "cube_shape" in z.files else None

    if stored is not None:
        if int(np.prod(stored)) != n_flux:
            raise ValueError(f"{os.path.basename(path)}: stored cube_shape {stored} "
                             f"does not match its {n_flux} flux elements.")
        cube_shape = stored
    else:
        if lenslet_grid is None:
            raise ValueError(f"{os.path.basename(path)} has no stored cube_shape; "
                             f"pass lenslet_grid=(ny, nx) to infer it.")
        ny, nx = (int(n) for n in lenslet_grid)
        if n_flux % (ny * nx):
            raise ValueError(f"{os.path.basename(path)} has {n_flux} flux elements, "
                             f"not divisible by the lenslet grid ny*nx = {ny}*{nx}. "
                             f"Check the lenslet grid for this mode, or regenerate "
                             f"the rectmat so it stores its cube_shape.")
        cube_shape = (n_flux // (ny * nx), ny, nx)
    return Rectmat(matrix, cube_shape, kind_from_shape(matrix), path=path)


def read_rectmat(path, lenslet_grid=None):
    """
    Load a rectmat (FITS, or legacy .npz).

    lenslet_grid=(ny, nx) is only used for legacy .npz files that do not
    store their cube shape.
    """
    if str(path).endswith(".npz"):
        return _read_npz(path, lenslet_grid)
    return _read_fits(path)


def convert_npz_rectmat(npz_path, out_path, wavelengths, header=None, kind=None,
                        lenslet_grid=None, monowave=None):
    """
    Convert a legacy .npz rectmat to the FITS format.

    wavelengths : array [um] or path to the old *_lams.fits / *_intp_lams.fits.
    """
    if isinstance(wavelengths, (str, os.PathLike)):
        wavelengths = fits.getdata(wavelengths)
    rect = _read_npz(npz_path, lenslet_grid)
    hdr = merge_header(fits.Header(), header)
    hdr.add_history(f"Converted from {os.path.basename(npz_path)}")
    return write_rectmat(out_path, rect.matrix, rect.cube_shape,
                         np.ravel(wavelengths), kind=kind or rect.kind,
                         header=hdr, monowave=monowave)
