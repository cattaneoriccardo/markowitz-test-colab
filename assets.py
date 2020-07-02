import numpy as np


class Assets:
    def __init__(self):
        self._rics_list = list()
        self._tot_return_per_asset_matrix = None
        self._avg_return_per_asset_matrix = None
        self._num_time_samples = None
        self._covariance_matrix = None
        self._is_covariance_matrix_stale = False
        return

    def add_or_modify_asset(self, ric_string, total_return_array, sanitize=True):
        if self._num_time_samples is None and len(total_return_array) == 0:
            raise Exception("Tried to add or modify an asset with 0 data/time points")
        if self._num_time_samples is not None and 0 < self._num_time_samples != len(total_return_array):
            raise Exception("Tried to add or modify an asset with too many or too few total return points")
        if self._num_time_samples is not None and self._num_time_samples < 0:
            raise Exception("Invariant violation, code is broken")

        # total return and average return
        total_return_nparray = np.array(total_return_array)
        if sanitize:
            total_return_nparray = np.nan_to_num(total_return_array)
        avg_return_array = np.average(total_return_nparray)
        total_return_nparray = np.reshape(total_return_nparray, (1, len(total_return_nparray)))
        avg_return_nparray = np.array(avg_return_array).reshape((1,1))

        # add or modify?
        unique_rics = set(self._rics_list)
        if ric_string in unique_rics:
            # old asset, modify
            ric_idx = self._rics_list.index(ric_string)
            self._tot_return_per_asset_matrix[ric_idx, :] = total_return_nparray
            self._avg_return_per_asset_matrix[ric_idx] = avg_return_nparray
        else:
            # new asset, add
            if len(unique_rics) == 0:
                self._rics_list.append(ric_string)
                self._num_time_samples = len(total_return_array)
                self._tot_return_per_asset_matrix = total_return_nparray
                self._avg_return_per_asset_matrix = avg_return_nparray
            else:
                self._rics_list.append(ric_string)
                self._tot_return_per_asset_matrix = np.vstack((self._tot_return_per_asset_matrix, total_return_nparray))
                self._avg_return_per_asset_matrix = np.vstack((self._avg_return_per_asset_matrix, avg_return_nparray))
        self._is_covariance_matrix_stale = True
        return

    # def remove_asset(self, ric_string):
    #     unique_rics = set(self._rics)
    #     if ric_string in unique_rics:
    #         if len(unique_rics) == 1:
    #             self._num_time_samples = 0
    #         self._total_return_per_asset.remove(self._rics.index(ric_string))
    #         self._rics.remove(ric_string)
    #         self._is_covariance_matrix_stale = True
    #     else:
    #         raise Exception("Tried to remove an asset not belonging to this instance object")

    def get_rics_list(self) -> list:
        return self._rics_list

    def get_total_returns(self):
        return self._tot_return_per_asset_matrix

    def get_all_assets_average_return(self):
        return self._avg_return_per_asset_matrix

    def get_covariance_matrix(self):
        return self._covariance_matrix

    def get_number_of_assets(self):
        return len(self._rics_list)

    def is_covariance_matrix_stale(self):
        return self._is_covariance_matrix_stale

    def compute_and_set_covariance_matrix(self):
        total_returns = self.get_total_returns()
        self._covariance_matrix = np.cov(total_returns)
        self._is_covariance_matrix_stale = False
