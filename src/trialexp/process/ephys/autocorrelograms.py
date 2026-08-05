from __future__ import annotations

import importlib.util
import warnings
import platform
from copy import deepcopy
from tqdm.auto import tqdm

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from threadpoolctl import threadpool_limits

import numpy as np

from spikeinterface.core import BaseSorting
from spikeinterface.core.job_tools import fix_job_kwargs, _shared_job_kwargs_doc
from spikeinterface.core.sortinganalyzer import (
    AnalyzerExtension,
    SortingAnalyzer,
    register_result_extension,
)
from spikeinterface.core.waveforms_extractor_backwards_compatibility import (
    MockWaveformExtractor,
)

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False


class ComputeAutoCorrelograms(AnalyzerExtension):
    """
    Compute auto and cross correlograms of unit spike times.

    Parameters
    ----------
    window_ms : float, default: 50.0
        The window around the spike to compute the correlation in ms. For example,
         if 50 ms, the correlations will be computed at lags -25 ms ... 25 ms.
    bin_ms : float, default: 1.0
        The bin size in ms. This determines the bin size over which to
        combine lags. For example, with a window size of -25 ms to 25 ms, and
        bin size 1 ms, the correlation will be binned as -25 ms, -24 ms, ...
    method : "auto" | "numpy" | "numba", default: "auto"
         If "auto" and numba is installed, numba is used, otherwise numpy is used.

    Returns
    -------
    correlogram : np.array
        Correlograms with shape (num_units, num_units, num_bins)
        The diagonal of the correlogram (e.g. correlogram[A, A, :])
        holds the unit auto correlograms. The off-diagonal elements
        are the cross-correlograms between units, where correlogram[A, B, :]
        and correlogram[B, A, :] represent cross-correlation between
        the same pair of units, applied in opposite directions,
        correlogram[A, B, :] = correlogram[B, A, ::-1].
    bins :  np.array
        The bin edges in ms

    Notes
    -----
    In the extracellular electrophysiology context, a correlogram
    is a visualisation of the results of a cross-correlation
    between two spike trains. The cross-correlation slides one spike train
    along another sample-by-sample, taking the correlation at each 'lag'. This results
    in a plot with 'lag' (i.e. time offset) on the x-axis and 'correlation'
    (i.e. how similar to two spike trains are) on the y-axis. In this
    implementation, the y-axis result is the 'counts' of spike matches per
    time bin (rather than a computer correlation or covariance).

    In the present implementation, a 'window' around spikes is first
    specified. For example, if a window of 100 ms is taken, we will
    take the correlation at lags from -50 ms to +50 ms around the spike peak.
    In theory, we can have as many lags as we have samples. Often, this
    visualisation is too high resolution and instead the lags are binned
    (e.g. -50 to -45 ms, ..., -5 to 0 ms, 0 to 5 ms, ...., 45 to 50 ms).
    When using counts as output, binning the lags involves adding up all counts across
    a range of lags.


    """

    extension_name = "autocorrelograms"
    depend_on = []
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = False

    def _set_params(self, window_ms: float = 50.0, bin_ms: float = 1.0, method: str = "auto"):
        params = dict(window_ms=window_ms, bin_ms=bin_ms, method=method)

        return params

    def _select_extension_data(self, unit_ids):
        # filter metrics dataframe
        unit_indices = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
        new_ccgs = self.data["ccgs"][unit_indices][:, unit_indices]
        new_bins = self.data["bins"]
        new_data = dict(ccgs=new_ccgs, bins=new_bins)
        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, censor_ms=None, verbose=False, **job_kwargs
    ):
        """
        When two units are merged, their cross-correlograms with other units become the sum
        of the previous cross-correlograms. More precisely, if units i and j get merged into
        unit k, then the new unit's cross-correlogram with any other unit l is:
            C_{k,l} = C_{i,l} + C_{j,l}
            C_{l,k} = C_{l,k} + C_{l,j}
        Here, we apply this formula to quickly compute correlograms for merged units.
        """

        can_apply_soft_method = True
        if censor_ms is not None:
            # if censor_ms has no effect, can apply "soft" method. Check if any spikes have been removed
            for new_unit_id, merge_unit_group in zip(new_unit_ids, merge_unit_groups):
                num_segments = new_sorting_analyzer.get_num_segments()
                for segment_index in range(num_segments):
                    merged_spike_train_length = len(
                        new_sorting_analyzer.sorting.get_unit_spike_train(new_unit_id, segment_index=segment_index)
                    )

                    old_spike_train_lengths = len(
                        np.concatenate(
                            [
                                self.sorting_analyzer.sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
                                for unit_id in merge_unit_group
                            ]
                        )
                    )

                    if merged_spike_train_length != old_spike_train_lengths:
                        can_apply_soft_method = False
                        break

        if can_apply_soft_method is False:
            new_ccgs, new_bins = _compute_correlograms_on_sorting(new_sorting_analyzer.sorting, **self.params)
            new_data = dict(ccgs=new_ccgs, bins=new_bins)
        else:
            # Make a transformation dict, which tells us how unit_indices from the
            # old to the new sorter are mapped.
            old_to_new_unit_index_map = {}
            for old_unit in self.sorting_analyzer.unit_ids:
                old_unit_index = self.sorting_analyzer.sorting.id_to_index(old_unit)
                unit_involved_in_merge = False
                for merge_unit_group, new_unit_id in zip(merge_unit_groups, new_unit_ids):
                    new_unit_index = new_sorting_analyzer.sorting.id_to_index(new_unit_id)
                    # check if the old_unit is involved in a merge
                    if old_unit in merge_unit_group:
                        # check if it is mapped to itself
                        if old_unit == new_unit_id:
                            old_to_new_unit_index_map[old_unit_index] = new_unit_index
                        # or to a unit_id outwith the old ones
                        elif new_unit_id not in self.sorting_analyzer.unit_ids:
                            if new_unit_index not in old_to_new_unit_index_map.values():
                                old_to_new_unit_index_map[old_unit_index] = new_unit_index
                        unit_involved_in_merge = True
                if unit_involved_in_merge is False:
                    old_to_new_unit_index_map[old_unit_index] = new_sorting_analyzer.sorting.id_to_index(old_unit)

            correlograms, new_bins = deepcopy(self.get_data())

            for new_unit_id, merge_unit_group in zip(new_unit_ids, merge_unit_groups):
                merge_unit_group_indices = self.sorting_analyzer.sorting.ids_to_indices(merge_unit_group)

                # Sum unit rows of the correlogram matrix: C_{k,l} = C_{i,l} + C_{j,l}
                # and place this sum in all indices from the merge group
                new_col = np.sum(correlograms[merge_unit_group_indices, :, :], axis=0)
                # correlograms[merge_unit_group_indices[0], :, :] = new_col
                correlograms[merge_unit_group_indices, :, :] = new_col
                # correlograms[merge_unit_group_indices[1:], :, :] = 0

                # Sum unit columns of the correlogram matrix: C_{l,k} = C_{l,i} + C_{l,j}
                # and put this sum in all indices from the merge group
                new_row = np.sum(correlograms[:, merge_unit_group_indices, :], axis=1)

                for merge_unit_group_index in merge_unit_group_indices:
                    correlograms[:, merge_unit_group_index, :] = new_row

            new_correlograms = np.zeros(
                (len(new_sorting_analyzer.unit_ids), len(new_sorting_analyzer.unit_ids), correlograms.shape[2])
            )
            for old_index_1, new_index_1 in old_to_new_unit_index_map.items():
                for old_index_2, new_index_2 in old_to_new_unit_index_map.items():
                    new_correlograms[new_index_1, new_index_2, :] = correlograms[old_index_1, old_index_2, :]
                    new_correlograms[new_index_2, new_index_1, :] = correlograms[old_index_2, old_index_1, :]

            new_data = dict(ccgs=new_correlograms, bins=new_bins)
        return new_data

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        # TODO: for now we just copy
        new_ccgs, new_bins = _compute_correlograms_on_sorting(new_sorting_analyzer.sorting, **self.params)
        new_data = dict(ccgs=new_ccgs, bins=new_bins)
        return new_data

    def _run(self, verbose=False):
        ccgs, bins = _compute_correlograms_on_sorting(self.sorting_analyzer.sorting, **self.params)
        self.data["ccgs"] = ccgs
        self.data["bins"] = bins

    def _get_data(self):
        return self.data["ccgs"], self.data["bins"]


register_result_extension(ComputeAutoCorrelograms)
compute_autocorrelograms_sorting_analyzer = ComputeAutoCorrelograms.function_factory()


def compute_autocorrelograms(
    sorting_analyzer_or_sorting,
    window_ms: float = 50.0,
    bin_ms: float = 1.0,
    method: str = "auto",
):
    """
    Compute correlograms using Numba or Numpy.
    See ComputeCorrelograms() for details.
    """
    if isinstance(sorting_analyzer_or_sorting, MockWaveformExtractor):
        sorting_analyzer_or_sorting = sorting_analyzer_or_sorting.sorting

    if isinstance(sorting_analyzer_or_sorting, SortingAnalyzer):
        return compute_autocorrelograms_sorting_analyzer(
            sorting_analyzer_or_sorting, window_ms=window_ms, bin_ms=bin_ms, method=method
        )
    else:
        return _compute_correlograms_on_sorting(
            sorting_analyzer_or_sorting, window_ms=window_ms, bin_ms=bin_ms, method=method
        )


compute_autocorrelograms.__doc__ = compute_autocorrelograms_sorting_analyzer.__doc__


def _make_bins(sorting, window_ms, bin_ms) -> tuple[np.ndarray, int, int]:
    """
    Create the bins for the correlogram, in samples.

    The autocorrelogram bins are centered around zero. Each bin
    increases in a positive / negative direction starting at zero.

    For example, given a window_ms of 50 ms and a bin_ms of
    5 ms, the bins in unit ms will be:
    [-25 to -20, ..., -5 to 0, 0 to 5, ..., 20 to 25].

    The window size will be clipped if not divisible by the bin size.

    Parameters
    ----------
    See ComputeCorrelograms() for parameters.

    Returns
    -------

    bins : np.ndarray
        The bins edges in ms
    window_size : int
        The window size in samples
    bin_size : int
        The bin size in samples

    """
    fs = sorting.sampling_frequency

    window_size = int(round(fs * window_ms / 2 * 1e-3))
    bin_size = int(round(fs * bin_ms * 1e-3))
    window_size -= window_size % bin_size
    num_bins = 2 * int(window_size / bin_size)
    assert num_bins >= 1, "Number of bins must be >= 1"

    bins = np.arange(-window_size, window_size + bin_size, bin_size) * 1e3 / fs

    return bins, window_size, bin_size


def _compute_num_bins(window_size, bin_size):
    """
    Internal function to compute number of bins, expects
    window_size and bin_size are already divisible. These are
    typically generated in `_make_bins()`.

    Returns
    -------
    num_bins : int
        The total number of bins to span the window, in samples
    half_num_bins : int
        Half the number of bins. The bins are an equal number
        of bins that look forward and backwards from zero, e.g.
        [..., -10 to -5, -5 to 0, 0 to 5, 5 to 10, ...]

    """
    num_half_bins = int(window_size // bin_size)
    num_bins = int(2 * num_half_bins)

    return num_bins, num_half_bins


def _compute_correlograms_on_sorting(sorting, window_ms, bin_ms, method="auto"):
    """
    Computes cross-correlograms from multiple units.

    Entry function to compute correlograms across all units in a `Sorting`
    object (i.e. spike trains at all determined offsets will be computed
    for each unit against every other unit).

    Parameters
    ----------
    sorting : Sorting
        A SpikeInterface Sorting object
    window_ms : float
            The window size over which to perform the cross-correlation, in ms
    bin_ms : float
        The size of which to bin lags, in ms.
    method : str
        To use "numpy" or "numba". "auto" will use numba if available,
        otherwise numpy.

    Returns
    -------
    correlograms : np.array
        A (num_units, num_units, num_bins) array where unit x unit correlation
        matrices are stacked at all determined time bins. Note the true
        correlation is not returned but instead the count of number of matches.
    bins : np.array
        The bins edges in ms
    """
    bins, window_size, bin_size = _make_bins(sorting, window_ms, bin_ms)

    correlograms = _compute_autocorrelograms_parallel(sorting, window_size, bin_size)

    return correlograms, bins

def _compute_autocorrelograms_parallel(sorting, window_size, bin_size):
    import concurrent.futures
    
    num_units = len(sorting.unit_ids)
    num_bins, num_half_bins = _compute_num_bins(window_size, bin_size)
    
    # Pre-fetch all spike trains to avoid pickling the whole sorting object
    # (or pass the sorting object if it's efficient to read in workers)
    spike_trains = [sorting.get_unit_spike_train(u) for u in sorting.unit_ids]
    
    # Run in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the helper function to each spike train
        results = list(executor.map(
            _compute_single_acg, 
            spike_trains, 
            [window_size]*num_units, 
            [bin_size]*num_units
        ))
    
    # Reassemble into the expected 3D shape (N, N, Bins)
    # The off-diagonals will remain 0
    correlograms = np.zeros((num_units, num_bins), dtype="int64")
    for i, acg in enumerate(results):
        correlograms[i, :] = acg
        
    return correlograms

def _compute_single_acg(spike_times, window_size, bin_size):
    """
    Simplified worker function for a single unit.
    Removes multi-index overhead.
    """
    # spike_times here belongs to a single cell
    num_bins, num_half_bins = _compute_num_bins(window_size, bin_size)
    acg = np.zeros(num_bins, dtype="int64")
    
    mask = np.ones_like(spike_times, dtype="bool")
    shift = 1
    
    while mask[:-shift].any():
        spike_diff = spike_times[shift:] - spike_times[:-shift]
        
        for sign in (-1, 1):
            spike_diff_b = (spike_diff * sign) // bin_size
            
            if sign == -1:
                mask[:-shift][spike_diff_b < -num_half_bins] = False
            else:
                mask[:-shift][spike_diff_b >= num_half_bins] = False
            
            m = mask[:-shift]
            
            # OPTIMIZATION: Direct 1D indexing instead of ravel_multi_index
            valid_bins = spike_diff_b[m] + num_half_bins
            
            if len(valid_bins) > 0:
                acg += np.bincount(valid_bins, minlength=num_bins)

            if sign == 1:
                mask[:-shift][spike_diff_b == num_half_bins] = True
                
        shift += 1
        
    return acg

