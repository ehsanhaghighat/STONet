"""
STONet: A Neural Operator for Modeling Solute Transport in Micro-Cracked Reservoirs

This code is part of the STONet repository: https://github.com/ehsanhaghighat/STONet

Citation:
@article{haghighat2024stonet,
  title={STONet: A neural operator for modeling solute transport in micro-cracked reservoirs},
  author={Haghighat, Ehsan and Adeli, Mohammad Hesan and Mousavi, S Mohammad and Juanes, Ruben},
  journal={arXiv preprint arXiv:2412.05576},
  year={2024}
}

Paper: https://arxiv.org/abs/2412.05576
"""


from abc import abstractmethod
from typing import Dict, Union, Any
from functools import reduce
import numpy
import pandas
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import logging


class BaseScaler:
    """ Base scaler class.
    """

    def __init__(self) -> None:
        self.min = 0.
        self.max = 0.
        self.mean = 0.
        self.std = 0.
        self.is_fitted = False

    def fit(self, x: Union[torch.Tensor, numpy.ndarray]):
        assert self.is_fitted is False, "The scaler is already fitted."
        if isinstance(x, numpy.ndarray):
            x_np = x  # type: numpy.ndarray
        elif isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()  # returns it as a numpy.ndarray
        else:
            raise ValueError("Unsupported type for x: {}".format(type(x)))
        self.min = x_np.min()
        self.max = x_np.max()
        self.mean = x_np.mean()
        self.std = x_np.std()
        self.is_fitted = True
        return self

    def transform(self, x):  # pylint: disable=R0201
        return x

    def inverse_transform(self, x):  # pylint: disable=R0201
        return x


class StandardScaler(BaseScaler):
    """ Scale data to zero mean and unit variance.
    """

    def transform(self, x):
        std = max(self.std, 1e-8)  # to avoid division by zero
        return (x - self.mean) / std

    def inverse_transform(self, x):
        return x * self.std + self.mean

class MinMaxScaler(BaseScaler):
    """ Scale data to a given range.
    """

    def __init__(self, feature_range=[0., 1.]) -> None:  # pylint: disable=W0102
        super().__init__()
        self.range = feature_range

    def transform(self, x):
        x_std = (x - self.min) / (self.max - self.min)
        return x_std * (self.range[1] - self.range[0]) + self.range[0]

    def inverse_transform(self, x):
        x_std = (x - self.range[0]) / (self.range[1] - self.range[0])
        return x_std * (self.max - self.min) + self.min

class DefaultScaler(StandardScaler):
    """ Default scaler.
    """

    def __init__(self) -> None:
        super().__init__()
        self.min = 0.
        self.max = 1.
        self.mean = 0.
        self.std = 1.
        self.is_fitted = True


class BaseDataModel(Dataset):
    def __init__(self):
        super().__init__()
        self.inputs = {}
        self.targets = {}
        self.weights = {}
        self.input_scaler = {}
        self.is_input_scaled = False
        self.target_scaler = {}
        self.is_target_scaled = False

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def get_data(self):
        return self.inputs, self.targets, self.weights

    def get_iterator(self, batch_size=64, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def to(self, device: torch.device):
        """ Move data to GPU.
        """
        for key in self.inputs:
            self.inputs[key] = self.inputs[key].to(device)
        for key in self.targets:
            self.targets[key] = self.targets[key].to(device)
        for key in self.weights:
            self.weights[key] = self.weights[key].to(device)
        self.sample_ids = self.sample_ids.to(device)
        self.batch_ids = [x.to(device) for x in self.batch_ids]

    def set_input_scaler(self, scaler: Dict[str, BaseScaler]):
        """ set input scaler for the model.
            If input scaler is set as a part of model, it is assumed data is not scaled before
            and the model will scale it.
        # Args:
            scaler: (Union[Dict, Any]) The scaler method to be used for scaling the inputs.
                    see `preprocessing.py` for more details.
        """
        req_methods = ['fit', 'transform', 'inverse_transform']
        for k in self.inputs:
            assert k in scaler, "The scaler for {} is not provided.".format(k)
            scaler_k = scaler[k]  # type: BaseScaler
            assert scaler_k.is_fitted, "The scaler for {} is not fitted.".format(k)
            assert all([hasattr(scaler_k, f) for f in req_methods]), \
                "The scaler must have the following methods: {}.".format(req_methods)
            self.input_scaler[k] = scaler_k

    def set_target_scaler(self, scaler: Dict[str, BaseScaler]):
        """ Set the output scaler for the model. See `set_input_scaler` for more details.
        """
        req_methods = ['fit', 'transform', 'inverse_transform']
        for k in self.targets:
            assert k in scaler, "The scaler for {} is not provided.".format(k)
            scaler_k = scaler[k]  # type: BaseScaler
            assert scaler_k.is_fitted, "The scaler for {} is not fitted.".format(k)
            assert all([hasattr(scaler_k, f) for f in req_methods]), \
                "The scaler must have the following methods: {}.".format(req_methods)
            assert self.target_scaler.get(k) is None, \
                "The target scaler for {} is already set.".format(k)
            self.target_scaler[k] = scaler_k

    def transform(self):
        """ Transform data using scalers.
        # Args:
            input_scaler: (Dictionary) input scalers.
                It should be stored as a part of the model to be retrieved in inference.
            target_scaler: (Dictionary) target scalers.
                It should be stored as a part of the model to be retrieved in inference.
        """
        assert not self.is_input_scaled, 'The input data is already scaled.'
        self.inputs = {k: self.input_scaler[k].transform(v) for k, v in self.inputs.items()}
        self.is_input_scaled = True
        assert not self.is_target_scaled, 'The target data is already scaled.'
        self.targets = {k: self.target_scaler[k].transform(v) for k, v in self.targets.items()}
        self.is_target_scaled = True

    def inverse_transform(self):
        """ Inverse transform data using scalers. """
        assert self.is_input_scaled, 'The input data is not scaled.'
        self.inputs = {k: self.input_scaler[k].inverse_transform(v) for k, v in self.inputs.items()}
        self.is_input_scaled = False
        assert self.is_target_scaled, 'The target data is not scaled.'
        self.targets = {k: self.target_scaler[k].inverse_transform(v) for k, v in self.targets.items()}
        self.is_target_scaled = False

    def transform_inputs(self, inputs: Dict[str, torch.Tensor]):
        """ Inverse transform data using scalers. """
        return {k: self.input_scaler[k].transform(v) for k, v in inputs.items()}

    def inverse_transform_outputs(self, outputs: Dict[str, torch.Tensor]):
        """ Inverse transform data using scalers. """
        return {k: self.target_scaler[k].inverse_transform(v) for k, v in outputs.items()}

    def to_csv(self, file_path: str, append: Dict = {}):  # pylint: disable=W0102
        """ Save data to csv file. """
        all_data = {**self.inputs, **self.targets, **append}
        all_data = {k: v.detach().cpu().numpy().flatten()
                    for k, v in all_data.items()}
        df = pandas.DataFrame(all_data)
        df.to_csv(file_path, index=False)

    def eval_input_scaler(self, scaler: Dict[str, BaseScaler] = {}) -> Dict[str, BaseScaler]:  # pylint: disable=W0102
        """ eval input scaler for the model.
            If input scaler is set as a part of model, it is assumed data is not scaled before
            and the model will scale it.
        # Args:
            scaler: (Union[Dict, Any]) The scaler method to be used for scaling the inputs.
                    see `preprocessing.py` for more details.
        """
        input_scaler = {}
        req_methods = ['fit', 'transform', 'inverse_transform']
        for k in self.inputs:
            scaler_k = scaler.get(k, DefaultScaler())
            assert all([hasattr(scaler_k, f) for f in req_methods]), \
                "The scaler must have the following methods: {}.".format(req_methods)
            input_scaler[k] = scaler_k.fit(self.inputs[k].detach().cpu().numpy())
        return input_scaler

    def eval_target_scaler(self, scaler: Dict[str, BaseScaler] = {}) -> Dict[str, BaseScaler]:  # pylint: disable=W0102
        """ Set the output scaler for the model. See `set_input_scaler` for more details.
        Defaults to `MinMaxScaler`.
        """
        target_scaler = {}
        req_methods = ['fit', 'transform', 'inverse_transform']
        for k in self.targets:
            scaler_k = scaler.get(k, DefaultScaler())
            assert all([hasattr(scaler_k, f) for f in req_methods]), \
                "The scaler must have the following methods: {}.".format(req_methods)
            target_scaler[k] = scaler_k.fit(self.targets[k].detach().cpu().numpy())
        return target_scaler

    def __str__(self):
        out = '\n'
        max_key_len = max([len(k) for k in self.inputs])
        for k, v in self.inputs.items():
            out += '{:>5}-Input: {:>KEY_LEN}: {}\n'.replace('KEY_LEN', str(max_key_len + 2)) \
                .format('-', k, v.shape)
        max_key_len = max([len(k) for k in self.targets])
        for k, v in self.targets.items():
            out += '{:>5}Target: {:>KEY_LEN}: {}\n'.replace('KEY_LEN', str(max_key_len + 2)) \
                .format('-', k, v.shape)
        max_key_len = max([len(k) for k in self.weights])
        for k, v in self.weights.items():
            out += '{:>5}Weight: {:>KEY_LEN}: {}\n'.replace('KEY_LEN', str(max_key_len + 2)) \
                .format('-', k, v.shape)
        return out



class MLPDataModel(BaseDataModel):
    """ Multi-layer perceptron data model.
    Prepares input data in the form of numpy arrays for training MLP model.
    # Args:
        data_set: Dictionary of numpy arrays associated with inputs and targets.
        input_dim: Dictionary of input dimensions.
        target_dim: Dictionary of output dimensions.
    """

    def __init__(self,
                 data_set: Dict[str, numpy.ndarray],
                 input_dim: Dict[str, int],
                 target_dim: Dict[str, int] = {},
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        MLPDataModel.validate_data(
            data_set, input_dim, target_dim)
        # set inputs
        num_samples = []
        self.inputs = {}
        for key in input_dim:
            self.inputs[key] = torch.tensor(data_set[key], dtype=dtype, requires_grad=True)
            num_samples.append(len(self.inputs[key]))
        # set targets
        self.targets = {}
        self.weights = {}
        for key in target_dim:
            self.targets[key] = torch.tensor(data_set[key], dtype=dtype)
            if 'weight:' + key in data_set:
                self.weights[key] = torch.tensor(data_set['weight:' + key], dtype=dtype)
            else:
                self.weights[key] = torch.tensor(data_set['default_weight:' + key], dtype=dtype)
            num_samples.append(len(self.targets[key]))
        # set num_samples
        assert len(set(num_samples)) == 1
        self.num_samples = num_samples[0]
        self.num_batch = 1
        self.sample_ids = torch.arange(self.num_samples, dtype=int)
        self.batch_ids = [self.sample_ids]

    def set_batch_size(self, batch_size: int = 128):
        self.num_batch = int(numpy.ceil(self.num_samples / batch_size))
        self.batch_ids = torch.tensor_split(self.sample_ids, self.num_batch)

    def shuffle(self):
        self.sample_ids = torch.randperm(self.num_samples, dtype=int)
        self.batch_ids = torch.tensor_split(self.sample_ids, self.num_batch)

    def __len__(self):
        return self.num_batch

    def __getitem__(self, index):
        ids = self.batch_ids[index]
        inputs = {key: value[ids] for key, value in self.inputs.items()}
        targets = {key: value[ids] for key, value in self.targets.items()}
        weights = {key: value[ids] for key, value in self.weights.items()}
        return inputs, targets, weights

    @staticmethod
    def validate_data(data_set: Dict[str, numpy.ndarray],
                      input_dim: Dict[str, int],
                      target_dim: Dict[str, int] = {}) -> bool:
        # check data size
        num_samples = []
        # check trunk inputs
        for key in input_dim:
            assert key in data_set, "Input key {} not found in data_set."
            assert len(data_set[key].shape) == 2, \
                "Expecting (num_samples, input_dim) input data."
            assert data_set[key].shape[-1] == input_dim[key], \
                "data_set[{key}] must have the same outer (feature) dimension as input_dim[{key}].".format(key=key)
            num_samples.append(len(data_set[key]))
        # set targets
        for key in target_dim:
            assert key in data_set, "Target key {} not found in data_set."
            assert data_set[key].shape[-1] == target_dim[key], \
                "data_set[{key}] must have the same outer (feature) dimension as output_dim[{key}].".format(key=key)
            num_samples.append(len(data_set[key]))
        # set num_samples
        assert len(set(num_samples)) == 1, "All inputs and targets must have the same number of samples."
        return True

    @staticmethod
    def prepare_dataframe(data_dict: Dict[str, numpy.ndarray],
                          input_dim: Dict[str, int],
                          target_dim: Dict[str, int] = {},
                          target_weight: Union[str, Dict[str, str]] = 'sample_weights',
                          dtype: numpy.dtype = numpy.dtype('float32')
                          ) -> Dict[str, numpy.ndarray]:
        """ Prepare data in the form of a dictionary of numpy arrays.
        NOTE: with dataframes, only 1D inputs are supported.
        # Args:
            df: pandas dataframe.
            input_dim: Dict[str, int] model inputs/dims.
            target_dim: Dict[str, int] model outputs/dims.
            target_weight: key/key-map in dataframe for sample weights.
        # Returns:
            data_set: Dictionary of numpy arrays associated with inputs and targets.
        """
        # prepare inputs
        data = {}
        for key, dim in input_dim.items():
            assert key in data_dict, "Key {} not found in dataframe.".format(key)
            assert dim == 1, "Key {} has wrong dimension.".format(key)
            data[key] = numpy.array(data_dict[key]).astype(dtype).reshape(-1, 1)
        # set outputs/sample weights
        # sample weights:
        # - are used to weight the loss function at each sample
        # - are normalized to sum to size_dataset
        for key, dim in target_dim.items():
            assert dim == 1, "Key {} has wrong dimension.".format(key)
            if key in data_dict:
                data[key] = numpy.array(data_dict[key]).astype(dtype).reshape(-1, 1)
            else:
                data[key] = numpy.zeros((len(data_dict[key]), 1), dtype=dtype)
                logging.warning("Key `{}` not found in dataframe.".format(key))
            default_weight_key = 'default_weight:' + key
            data[default_weight_key] = numpy.ones((len(data_dict[key]), 1), dtype=dtype)
            weight_key = 'weight:' + key
            if isinstance(target_weight, str) and (target_weight in data_dict):
                data[weight_key] = numpy.array(data_dict[target_weight]).astype(dtype).reshape(-1, 1)
            elif isinstance(target_weight, dict) and (key in target_weight):
                data[weight_key] = numpy.array(data_dict[target_weight[key]]).astype(dtype).reshape(-1, 1)
            else:
                logging.info('No weight is provided for output key {}.'.format(key))
        return data


class DeepONetDataModel(BaseDataModel):
    """ Operator network data model.
    # Args:
        data_set: Dictionary containing numpy arrays associated inputs and targets.
        trunk_input_dim: Dictionary of trunk input dimensions.
        branch_input_dim: Dictionary of branch input dimensions.
        output_dim: Dictionary of output dimensions.
    """

    def __init__(self,
                 data_set: Dict[str, numpy.ndarray],
                 trunk_input_dim: Dict[str, int],
                 branch_input_dim: Dict[str, int],
                 target_dim: Dict[str, int] = {},
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        DeepONetDataModel.validate_data(
            data_set, trunk_input_dim, branch_input_dim, target_dim)
        # set inputs
        num_samples, num_sensors = [], []
        self.inputs = {}
        # set trunk inputs
        for key in trunk_input_dim:
            self.inputs[key] = torch.tensor(data_set[key], dtype=dtype, requires_grad=True)
            num_sensors.append(self.inputs[key].shape[0])
            num_samples.append(self.inputs[key].shape[1])
        # set branch inputs
        for key in branch_input_dim:
            self.inputs[key] = torch.tensor(data_set[key], dtype=dtype, requires_grad=True)
            num_sensors.append(self.inputs[key].shape[0])
            num_samples.append(self.inputs[key].shape[1])
        # set targets
        self.targets = {}
        self.weights = {}
        for key in target_dim:
            self.targets[key] = torch.tensor(data_set[key], dtype=dtype)
            if 'weight:' + key in data_set:
                self.weights[key] = torch.tensor(data_set['weight:' + key], dtype=dtype)
            else:
                self.weights[key] = torch.tensor(data_set['default_weight:' + key], dtype=dtype)
            num_sensors.append(self.targets[key].shape[0])
        # set num_samples / num_sensors
        assert len(set(num_samples)) == 1
        self.num_samples_per_sensor = num_samples[0]
        logging.info("Number of samples: {}".format(self.num_samples_per_sensor))
        assert len(set(num_sensors)) == 1
        self.num_sensors = num_sensors[0]
        logging.info("Number of sensors: {}".format(self.num_sensors))
        self.total_num_samples = self.num_sensors * self.num_samples_per_sensor
        logging.info("Total number of samples: {}".format(self.total_num_samples))
        self.num_batch = 1
        self.sample_ids = torch.arange(self.num_sensors, dtype=int)
        self.batch_ids = [self.sample_ids]

    def set_batch_size(self, batch_size: int = 2**3):
        self.num_batch = int(numpy.ceil(self.num_sensors / batch_size))
        self.batch_ids = torch.tensor_split(self.sample_ids, self.num_batch)

    def shuffle(self):
        self.sample_ids = torch.randperm(self.num_sensors, dtype=int)
        self.batch_ids = torch.tensor_split(self.sample_ids, self.num_batch)

    def __len__(self):
        return self.num_batch

    def __getitem__(self, index):
        ids = self.batch_ids[index]
        inputs = {key: value[ids] for key, value in self.inputs.items()}
        targets = {key: value[ids] for key, value in self.targets.items()}
        weights = {key: value[ids] for key, value in self.weights.items()}
        return inputs, targets, weights

    @staticmethod
    def validate_data(data_set: Dict[str, numpy.ndarray],
                      trunk_input_dim: Dict[str, int],
                      branch_input_dim: Dict[str, int],
                      target_dim: Dict[str, int] = {}) -> bool:
        # check data size
        num_samples, num_sensors = [], []
        # check trunk inputs
        trunk_branch_input_dim = {**trunk_input_dim, **branch_input_dim}
        for key in trunk_branch_input_dim:
            assert key in data_set, "Input key {} not found in data_set."
            assert len(data_set[key].shape) == 3, \
                "Expecting (num_sensors, num_samples, input_dim) input data."
            assert data_set[key].shape[-1] == trunk_branch_input_dim[key], \
                "data_set[{key}] must have the same outer (feature) dimension as input_dim[{key}].".format(key=key)
            num_sensors.append(data_set[key].shape[0])
            num_samples.append(data_set[key].shape[1])
        # set targets
        for key in target_dim:
            assert key in data_set, "Target key {} not found in data_set."
            assert data_set[key].shape[-1] == target_dim[key], \
                "data_set[{key}] must have the same outer (feature) dimension as output_dim[{key}].".format(key=key)
            num_sensors.append(data_set[key].shape[0])
            num_samples.append(data_set[key].shape[1])
        # check sample size / sensor size
        assert len(set(num_sensors)) == 1, \
            "All inputs and targets must have the same number of sensors."
        assert len(set(num_samples)) == 1, \
            "All inputs and targets must have the same number of samples."
        return True

    @staticmethod
    def prepare_dataframe(data_dict: Dict[str, numpy.ndarray],
                          trunk_input_dim: Dict[str, int],
                          branch_input_dim: Dict[str, int],
                          target_dim: Dict[str, int] = {},
                          target_weight: Union[str, Dict[str, str]] = 'sample_weights',
                          batch_by: str = 'sample',
                          dtype: numpy.dtype = numpy.dtype('float32')
                          ) -> Dict[str, numpy.ndarray]:
        """ Prepare data in the form of a dictionary of numpy arrays.
        NOTE: with dataframes, only 1D inputs are supported.
        The expcted data format is:
            sensor 1, sensor 2, ..., feature 1, feature 2, ...
            batch 1
            0, 0, ..., any, any, ...
            0, 0, ..., any, any, ...
            0, 0, ..., any, any, ...
            batch 2
            1, 2, ..., any, any, ...
            1, 2, ..., any, any, ...
            1, 2, ..., any, any, ...
            etc
        For each batch, the value of sensors should be the same.
        # Args:
            dataframe: pandas dataframe.
            trunk_input_dim: Dictionary of trunk input dimensions.
            branch_input_dim: Dictionary of branch input dimensions.
            output_dim: Dictionary of output dimensions.
            output_weight: key/key-map in dataframe for sample weights.
        # Returns:
            data_set: Dictionary of numpy arrays associated with inputs and targets.
        """
        size_dataset = set({len(v) for v in data_dict.values()})
        assert len(size_dataset) == 1, "All inputs and targets must have the same number of samples."
        size_dataset = size_dataset.pop()
        # We want to split data by sensor values.
        sensor_ids = []
        if batch_by in data_dict:
            ids = numpy.where(numpy.diff(data_dict[batch_by]) != 0)[0] + 1
            sensor_ids.append(
                numpy.concatenate([[0], ids, [len(data_dict[batch_by])]])
            )
        else:
            raise ValueError("Unknown batch_by: {}".format(batch_by))
        batch_ids = reduce(numpy.union1d, sensor_ids)  # unique and sorted
        assert batch_ids[-1] == size_dataset, "Last batch id must be the last row."
        logging.info("Batch ids: {}...{}".format(batch_ids[:5], batch_ids[-5:]))
        batch_sizes = numpy.diff(batch_ids)
        batch_size_set = set(batch_sizes)
        logging.info("[batch-size:num-sample]: {}".format(
            {idx: (batch_sizes == idx).sum() for idx in batch_size_set}))
        if len(batch_size_set) > 1:
            logging.warning("Batch sizes are not equal. Zero-padding will be used.")
        input_dim = {**trunk_input_dim, **branch_input_dim}
        data = {}
        for key, dim in input_dim.items():
            assert key in data_dict, f"Key `{key}` not found in dataframe."
            assert dim == 1, f"Key `{key}` has wrong dimension."
            data[key] = numpy.zeros((len(batch_sizes), batch_sizes.max(), 1), dtype=dtype)
            for i, (start, end) in enumerate(zip(batch_ids[:-1], batch_ids[1:])):
                data[key][i, :batch_sizes[i], 0] = data_dict[key][start:end]
        # set outputs/sample weights
        # sample weights:
        # - are used to weight the loss function at each sample
        # - are normalized to sum to size_dataset
        for key, dim in target_dim.items():
            assert dim == 1, f"Key `{key}` has wrong dimension."
            data[key] = numpy.zeros((len(batch_sizes), batch_sizes.max(), 1), dtype=dtype)
            default_weight_key = 'default_weight:' + key
            data[default_weight_key] = numpy.zeros((len(batch_sizes), batch_sizes.max(), 1), dtype=dtype)
            main_sample_weights = numpy.zeros((len(batch_sizes), batch_sizes.max(), 1), dtype=dtype)
            for i, (start, end) in enumerate(zip(batch_ids[:-1], batch_ids[1:])):
                if key in data_dict:
                    data[key][i, :batch_sizes[i], 0] = data_dict[key][start:end]
                elif i == 0:
                    logging.warning(f"Key `{key}` not found in dataframe.")
                # setting default sample weights
                data[default_weight_key][i, :batch_sizes[i], 0] = numpy.ones(batch_sizes[i])
                # normalize sample weights to sum to size_dataset
                if isinstance(target_weight, str) and (target_weight in data_dict):
                    if i == 0:
                        logging.info(f"Using default sample weights from key: {target_weight}")
                    sample_weights = numpy.array(data_dict[target_weight][start:end])
                    norm_sample_weights = sum(data_dict[target_weight]) / size_dataset
                elif isinstance(target_weight, dict) and (key in target_weight):
                    if i == 0:
                        logging.info(f"Using the provided output_weights from key: {target_weight[key]}")
                    sample_weights = numpy.array(data_dict[target_weight[key]][start:end])
                    norm_sample_weights = sum(data_dict[target_weight[key]]) / size_dataset
                else:
                    continue
                assert min(sample_weights) >= 0.0, "Sample weights must be non-negative."
                main_sample_weights[i, :batch_sizes[i], 0] = sample_weights / norm_sample_weights
            if main_sample_weights.sum() > 0.0:
                data['weight:' + key] = main_sample_weights
        return data


if __name__ == "__main__":
    pass
