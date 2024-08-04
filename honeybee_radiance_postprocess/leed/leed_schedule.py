"""Module for dynamic LEED schedules."""
from typing import Tuple
import numpy as np

from ..results.annual_daylight import AnnualDaylight
from ..util import filter_array


def shd_trans_schedule_descending(
        results: AnnualDaylight, grid_info, light_paths, shade_transmittances, occ_mask,
        states_schedule, fail_to_comply
        ) -> Tuple[dict, dict]:
    grid_count = grid_info['count']
    full_direct = []
    full_thresh = []
    full_shd_trans_array = []
    for light_path in light_paths:
        array = results._get_array(grid_info, light_path, res_type="direct")
        array = np.apply_along_axis(filter_array, 1, array, occ_mask)
        full_direct.append(array)
        full_thresh.append((array >= 1000).sum(axis=0))
        full_shd_trans_array.append(shade_transmittances[light_path][1])

    # Sum the array element-wise.
    # This array is the sum of all direct illuminance without shade
    # transmittance.
    full_direct_sum = sum(full_direct)

    # Create base list of shading combinations (all set to 1).
    # We will replace the 1s later.
    combinations = [
        {light_path: 1 for light_path in light_paths}
        for i in range(full_direct_sum.shape[1])
    ]

    # Find the percentage of floor area >= 1000 lux.
    # This array is the percentage for each hour (axis=0).
    direct_pct_above = (full_direct_sum >= 1000).sum(axis=0) / grid_count

    # Find the indices where the percentage of floor area is > 2%.
    # This array is the problematic hours.
    above_2_indices = np.where(direct_pct_above > 0.02)[0]

    # Use the indices to get the relevant hours.
    direct_sum = np.take(full_direct_sum, above_2_indices, axis=1)

    # Use the indices to get the relevant hours.
    direct = np.take(full_direct, above_2_indices, axis=2)

    # Use the indices to get the relevant hours.
    thresh = np.take(full_thresh, above_2_indices, axis=1)

    # Sort and get indices. Negate the array to get descending order.
    # Descending order puts the "highest offender" light path first.
    sort_thresh = np.argsort(-thresh, axis=0).transpose()

    _combinations = []
    _combinations.insert(
        0, (np.arange(full_direct_sum.shape[1]), combinations)
    )

    if np.any(above_2_indices):
        # There are hours where the percentage of floor area is > 2%.
        for idx, lp in enumerate(light_paths):
            # Take column. For each iteration it will take the next column
            # in descending order, i.e., the "highest offender" is the first
            # column.
            sort_indices = np.take(sort_thresh, idx, axis=1)

            # Map light path identifiers to indices.
            light_path_ids = np.take(light_paths, sort_indices)

            # Map shade transmittance to indices.
            shd_trans_array = np.take(full_shd_trans_array, sort_indices)

            # Create combination for the subset.
            _subset_combination = [
                {light_path: _shd_trans} for light_path, _shd_trans in
                zip(light_path_ids, shd_trans_array)
            ]
            _combinations.insert(0, (above_2_indices, _subset_combination))

            # Take the values from each array by indexing.
            direct_array = \
                direct[sort_indices, :, range(len(sort_indices))].transpose()

            # Subtract the illuminance values.
            direct_sum = direct_sum - (direct_array * (1 - shd_trans_array))

            # Find the percentage of floor area >= 1000 lux.
            direct_pct_above = (direct_sum >= 1000).sum(axis=0) / grid_count

            # Find the indices where the percentage of floor area is > 2%.
            above_2_indices = np.where(direct_pct_above > 0.02)[0]

            # Break if there are no hours above 2%.
            if not np.any(above_2_indices):
                break

            # Update variables for the next iteration.
            direct_sum = np.take(direct_sum, above_2_indices, axis=1)
            direct = np.take(direct, above_2_indices, axis=2)
            thresh = np.take(thresh, above_2_indices, axis=1)
            sort_thresh = np.take(sort_thresh, above_2_indices, axis=0)

        if np.any(above_2_indices):
            # There are hours not complying with the 2% rule.
            previous_indices = []
            previous_combination = []
            grid_comply = []
            # Merge the combinations from the iterations of the subsets.
            for i, subset in enumerate(_combinations):
                if i == 0:
                    previous_indices = subset[0]
                else:
                    _indices = subset[0]
                    grid_comply = []
                    for _pr_idx in previous_indices:
                        grid_comply.append(_indices[_pr_idx])
                    previous_indices = grid_comply
            # Convert indices to sun up hours indices.
            filter_indices = np.where(occ_mask.astype(bool))[0]
            grid_comply = [filter_indices[_gc] for _gc in grid_comply]
            grid_comply = np.array(results.sun_up_hours)[grid_comply]
            fail_to_comply[grid_info['name']] = \
                [int(hoy) for hoy in grid_comply]

        previous_indices = None
        previous_combination = None
        # Merge the combinations from the iterations of the subsets.
        for i, subset in enumerate(_combinations):
            if i == 0:
                previous_indices, previous_combination = subset
            else:
                _indices, _combination = subset
                for _pr_idx, _pr_comb in \
                        zip(previous_indices, previous_combination):
                    for light_path, _shd_trans in _pr_comb.items():
                        _combination[_pr_idx][light_path] = _shd_trans
                previous_indices = _indices
                previous_combination = _combination

        combinations = _combination

    # Merge the combinations of dicts.
    for combination in combinations:
        for light_path, shd_trans in combination.items():
            if light_path != "__static_apertures__":
                states_schedule[light_path].append(shd_trans)

    return states_schedule, fail_to_comply


def states_schedule_descending(
        results: AnnualDaylight, grid_info, light_paths, occ_mask,
        states_schedule, fail_to_comply
        ) -> Tuple[dict, dict]:
    grid_count = grid_info['count']
    full_direct = []
    full_thresh = []
    full_direct_blinds = []
    for light_path in light_paths:
        array = results._get_array(grid_info, light_path, state=0, res_type="direct")
        array = np.apply_along_axis(filter_array, 1, array, occ_mask)
        full_direct.append(array)
        full_thresh.append((array >= 1000).sum(axis=0))

        array = results._get_array(grid_info, light_path, state=1, res_type="direct")
        array = np.apply_along_axis(filter_array, 1, array, occ_mask)
        full_direct_blinds.append(array)

    # Sum the array element-wise.
    # This array is the sum of all direct illuminance without shade
    # transmittance.
    full_direct_sum = sum(full_direct)

    # Create base list of shading combinations (all set to 0).
    # We will replace the 0s later.
    combinations = [
        {light_path: 0 for light_path in light_paths}
        for i in range(full_direct_sum.shape[1])
    ]

    # Find the percentage of floor area >= 1000 lux.
    # This array is the percentage for each hour (axis=0).
    direct_pct_above = (full_direct_sum >= 1000).sum(axis=0) / grid_count

    # Find the indices where the percentage of floor area is > 2%.
    # This array is the problematic hours.
    above_2_indices = np.where(direct_pct_above > 0.02)[0]

    # Use the indices to get the relevant hours.
    direct_sum = np.take(full_direct_sum, above_2_indices, axis=1)

    # Use the indices to get the relevant hours.
    direct = np.take(full_direct, above_2_indices, axis=2)

    # Use the indices to get the relevant hours.
    direct_blinds = np.take(full_direct_blinds, above_2_indices, axis=2)

    # Use the indices to get the relevant hours.
    thresh = np.take(full_thresh, above_2_indices, axis=1)

    # Sort and get indices. Negate the array to get descending order.
    # Descending order puts the "highest offender" light path first.
    sort_thresh = np.argsort(-thresh, axis=0).transpose()

    _combinations = []
    _combinations.insert(
        0, (np.arange(full_direct_sum.shape[1]), combinations)
    )

    if np.any(above_2_indices):
        # There are hours where the percentage of floor area is > 2%.
        for idx, lp in enumerate(light_paths):
            # Take column. For each iteration it will take the next column
            # in descending order, i.e., the "highest offender" is the first
            # column.
            sort_indices = np.take(sort_thresh, idx, axis=1)

            # Map light path identifiers to indices.
            light_path_ids = np.take(light_paths, sort_indices)

            # Create combination for the subset.
            _subset_combination = [
                {light_path: 1} for light_path in light_path_ids
            ]
            _combinations.insert(0, (above_2_indices, _subset_combination))

            # Take the values from each array by indexing.
            direct_array = \
                direct[sort_indices, :, range(len(sort_indices))].transpose()

            direct_array = direct_blinds[sort_indices, :, range(len(sort_indices))].transpose()

            # Subtract the illuminance values.
            direct_sum = direct_sum - (direct_array * (1 - shd_trans_array))

            # Find the percentage of floor area >= 1000 lux.
            direct_pct_above = (direct_sum >= 1000).sum(axis=0) / grid_count

            # Find the indices where the percentage of floor area is > 2%.
            above_2_indices = np.where(direct_pct_above > 0.02)[0]
            print(above_2_indices)
            # Break if there are no hours above 2%.
            if not np.any(above_2_indices):
                break

            # Update variables for the next iteration.
            direct_sum = np.take(direct_sum, above_2_indices, axis=1)
            direct = np.take(direct, above_2_indices, axis=2)
            thresh = np.take(thresh, above_2_indices, axis=1)
            sort_thresh = np.take(sort_thresh, above_2_indices, axis=0)

        if np.any(above_2_indices):
            # There are hours not complying with the 2% rule.
            previous_indices = []
            previous_combination = []
            grid_comply = []
            # Merge the combinations from the iterations of the subsets.
            for i, subset in enumerate(_combinations):
                if i == 0:
                    previous_indices = subset[0]
                else:
                    _indices = subset[0]
                    grid_comply = []
                    for _pr_idx in previous_indices:
                        grid_comply.append(_indices[_pr_idx])
                    previous_indices = grid_comply
            # Convert indices to sun up hours indices.
            filter_indices = np.where(occ_mask.astype(bool))[0]
            grid_comply = [filter_indices[_gc] for _gc in grid_comply]
            grid_comply = np.array(results.sun_up_hours)[grid_comply]
            fail_to_comply[grid_info['name']] = \
                [int(hoy) for hoy in grid_comply]

        previous_indices = None
        previous_combination = None
        # Merge the combinations from the iterations of the subsets.
        for i, subset in enumerate(_combinations):
            if i == 0:
                previous_indices, previous_combination = subset
            else:
                _indices, _combination = subset
                for _pr_idx, _pr_comb in \
                        zip(previous_indices, previous_combination):
                    for light_path, _shd_trans in _pr_comb.items():
                        _combination[_pr_idx][light_path] = _shd_trans
                previous_indices = _indices
                previous_combination = _combination

        combinations = _combination

    # Merge the combinations of dicts.
    for combination in combinations:
        for light_path, shd_trans in combination.items():
            if light_path != "__static_apertures__":
                states_schedule[light_path].append(shd_trans)

    return states_schedule, fail_to_comply
