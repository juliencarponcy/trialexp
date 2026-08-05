"""
Compute extremum channel for each unit in a SortingAnalyzer.
"""

from __future__ import annotations

import pandas as pd

from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension
from spikeinterface.core.template_tools import get_template_extremum_channel


class ComputeExtremumChannel(AnalyzerExtension):
    """
    Compute the extremum channel for each unit.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object
    peak_sign : {"neg", "pos"}, default: "neg"
        Whether to use the positive ("pos") or negative ("neg") peaks to estimate extremum channels.

    Returns
    -------
    extremum_channels : pd.Series
        Series with unit_id as index and extremum channel_id as values

    Examples
    --------
    >>> analyzer.compute("extremum_channel")
    >>> extremum_channels = analyzer.get_extension("extremum_channel").get_data()
    """

    extension_name = "extremum_channel"
    depend_on = ["templates"]
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = False
    need_backward_compatibility_on_load = False

    def _set_params(self, peak_sign="neg"):
        params = dict(peak_sign=peak_sign)
        return params

    def _select_extension_data(self, unit_ids):
        new_extremum_channels = self.data["extremum_channels"].loc[unit_ids]
        return dict(extremum_channels=new_extremum_channels)

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        peak_sign = self.params["peak_sign"]
        old_extremum_channels = self.data["extremum_channels"]

        all_unit_ids = new_sorting_analyzer.unit_ids
        not_new_ids = all_unit_ids[~pd.Series(all_unit_ids).isin(new_unit_ids).values]

        # Compute extremum channels for new units
        new_extremum_channels = get_template_extremum_channel(
            new_sorting_analyzer, peak_sign=peak_sign, outputs="id"
        )

        # Combine old and new
        extremum_channels = pd.Series(index=all_unit_ids, dtype=old_extremum_channels.dtype)
        extremum_channels.loc[not_new_ids] = old_extremum_channels.loc[not_new_ids]
        extremum_channels.loc[new_unit_ids] = pd.Series(new_extremum_channels)[new_unit_ids]

        new_data = dict(extremum_channels=extremum_channels)
        return new_data

    def _run(self, verbose=False):
        peak_sign = self.params["peak_sign"]

        # Compute extremum channel for each unit
        extremum_channel_dict = get_template_extremum_channel(
            self.sorting_analyzer, peak_sign=peak_sign, outputs="index"
        )

        # Convert to pandas Series
        extremum_channels = pd.Series(extremum_channel_dict, name="extremum_channel_id")
        extremum_channels.index.name = "unit_id"

        self.data["extremum_channels"] = extremum_channels

    def _get_data(self):
        return self.data["extremum_channels"].values


register_result_extension(ComputeExtremumChannel)
compute_extremum_channel = ComputeExtremumChannel.function_factory()
