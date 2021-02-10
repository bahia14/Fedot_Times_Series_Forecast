from copy import deepcopy
from os.path import join
from threading import Thread, Lock
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze.morris import analyze as morris_analyze
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample import saltelli
from SALib.sample.latin import sample as lhc_sample
from SALib.sample.morris import sample as morris_sample
from sklearn.metrics import mean_squared_error

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.models.model_template import extract_model_params
from fedot.sensitivity.node_sensitivity import NodeAnalyzeApproach
from fedot.sensitivity.sensitivity_utils import \
    model_params_with_bounds_by_model_name, INTEGER_PARAMS


class ModelAnalyze(NodeAnalyzeApproach):
    lock = Lock()

    def __init__(self, chain: Chain, train_data, test_data: InputData, path_to_save=None):
        super(ModelAnalyze, self).__init__(chain, train_data, test_data, path_to_save)
        self.model_params = None
        self.model_type = None
        self.problem = None
        self.analyze_method = None
        self.sample_method = None
        self.manager_dict = {}

    def analyze(self, node_id: int,
                sa_method: str = 'sobol',
                sample_method: str = 'saltelli',
                sample_size: int = 1,
                is_oat: bool = True) -> Union[List[dict], float]:

        # check whether the chain is fitted
        if not self._chain.fitted_on_data:
            self._chain.fit(self._train_data)

        # define methods
        self.analyze_method = analyze_method_by_name.get(sa_method)
        self.sample_method = sample_method_by_name.get(sample_method)

        # create problem
        self.model_type: str = self._chain.nodes[node_id].model.model_type
        self.model_params = model_params_with_bounds_by_model_name.get(self.model_type)
        self.problem = _create_problem(self.model_params)

        # sample
        samples = self.sample(sample_size)
        converted_samples = _convert_sample_to_dict(self.problem, samples)
        clean_sample_variables(converted_samples)

        response_matrix = self.get_model_response_matrix(samples, node_id)
        indices = self.analyze_method(self.problem, response_matrix)
        converted_to_json_indices = convert_results_to_json(problem=self.problem,
                                                            si=indices)

        if is_oat:
            self._one_at_a_time_analyze(node_id=node_id,
                                        samples=samples)

        return [converted_to_json_indices]

    def sample(self, *args) -> Union[Union[List[Chain], Chain], np.array]:
        sample_size = args[0]
        samples = self.sample_method(self.problem, num_of_samples=sample_size)

        return samples

    def get_model_response_matrix(self, samples: List[dict], node_id: int):
        model_response_matrix = []
        for sample in samples:
            chain = deepcopy(self._chain)
            chain.nodes[node_id].custom_params = sample

            chain.fit(self._train_data)
            prediction = chain.predict(self._test_data)
            model_response_matrix.append(mean_squared_error(y_true=self._test_data.target,
                                                            y_pred=prediction.predict))

        return np.array(model_response_matrix)

    def worker(self, params: List[dict], samples, node_id):
        # default values of param & loss
        param_name = list(params[0].keys())[0]
        default_param_value = extract_model_params(self._chain.nodes[node_id]).get(param_name)

        # percentage ratio
        samples = (samples - default_param_value) / default_param_value
        response_matrix = self.get_model_response_matrix(params, node_id)
        response_matrix = (response_matrix - np.mean(response_matrix)) / (max(response_matrix) - min(response_matrix))

        ModelAnalyze.lock.acquire()
        self.manager_dict[f'{param_name}'] = [samples.reshape(1, -1)[0], response_matrix]
        ModelAnalyze.lock.release()

    def _visualize(self, data: dict):
        x_ticks_param = list()
        x_ticks_loss = list()
        for param in data.keys():
            x_ticks_param.append(param)
            x_ticks_loss.append(f'{param}_loss')
        param_values_data = list()
        losses_data = list()
        for value in data.values():
            param_values_data.append(value[0])
            losses_data.append(value[1])

        fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))
        ax1.boxplot(param_values_data)
        ax2.boxplot(losses_data)
        ax1.set_title('param')
        ax1.set_xticks(range(1, len(x_ticks_param) + 1))
        ax1.set_xticklabels(x_ticks_param)
        ax2.set_title('loss')
        ax2.set_xticks(range(1, len(x_ticks_loss) + 1))
        ax2.set_xticklabels(x_ticks_loss)

        plt.savefig(join(self._path_to_save, f'{self.model_type}_hp_sa.jpg'))

    def _one_at_a_time_analyze(self, node_id, samples: np.array):
        transposed_samples = samples.T

        one_at_a_time_params = []
        for index, param in enumerate(self.problem['names']):
            samples_per_param = [{param: value} for value in transposed_samples[index]]
            clean_sample_variables(samples_per_param)
            one_at_a_time_params.append(samples_per_param)

        jobs = [Thread(target=self.worker,
                       args=(params, transposed_samples[index], node_id))
                for index, params in enumerate(one_at_a_time_params)]

        for job in jobs:
            job.start()

        for job in jobs:
            job.join()

        self._visualize(data=self.manager_dict)


def sobol_method(problem, samples, model_response) -> dict:
    indices = sobol_analyze(problem, model_response, print_to_console=False)

    return indices


def morris_method(problem, samples, model_response) -> dict:
    indices = morris_analyze(problem=problem,
                             X=samples, Y=model_response, print_to_console=False)

    return indices


def make_saltelly_sample(problem, num_of_samples=100):
    samples = saltelli.sample(problem, num_of_samples)

    return samples


def make_moris_sample(problem, num_of_samples=100):
    samples = morris_sample(problem, num_of_samples, num_levels=4)

    return samples


def make_latin_hypercube_sample(problem, num_of_samples=100):
    samples = lhc_sample(problem, num_of_samples)

    return samples


def clean_sample_variables(samples: List[dict]):
    """Make integer values for params if necessary"""
    for sample in samples:
        for key, value in sample.items():
            if key in INTEGER_PARAMS:
                sample[key] = int(value)

    return samples


def _create_problem(params: dict):
    problem = {
        'num_vars': len(params),
        'names': list(params.keys()),
        'bounds': list()
    }

    for key, bounds in params.items():
        if bounds[0] is not str:
            bounds = list(bounds)
            problem['bounds'].append([bounds[0], bounds[-1]])
        else:
            problem['bounds'].append(bounds)
    return problem


def _convert_sample_to_dict(problem, samples) -> List[dict]:
    converted_samples = []
    names_of_params = problem['names']
    for sample in samples:
        new_params = {}
        for index, value in enumerate(sample):
            new_params[names_of_params[index]] = value
        converted_samples.append(new_params)

    return converted_samples


def convert_results_to_json(problem: dict, si: dict):
    sobol_indices = []
    for index in range(problem['num_vars']):
        var_indices = {f"{problem['names'][index]}": {
            'S1': list(si['S1'])[index],
            'S1_conf': list(si['S1_conf'])[index],
            'ST': list(si['ST'])[index],
            'ST_conf': list(si['ST_conf'])[index],
        }}
        sobol_indices.append(var_indices)

    data = {
        'problem': {
            'num_vars': problem['num_vars'],
            'names': problem['names'],
            'bounds': problem['bounds']

        },
        'sobol_indices': sobol_indices
    }

    return data


analyze_method_by_name = {
    'sobol': sobol_method,
    'morris': morris_method,
}

sample_method_by_name = {
    'saltelli': make_saltelly_sample,
    'morris': make_saltelly_sample,
    'sobol_sequence': None,
    'latin_hyper_cube': make_latin_hypercube_sample,

}
