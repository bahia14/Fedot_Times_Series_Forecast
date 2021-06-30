from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta

import numpy as np

from fedot.core.log import Log, default_log
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.tune.time_series import ts_cross_validation
from fedot.core.validation.tune.simple import fit_predict_one_fold


class HyperoptTuner(ABC):
    """
    Base class for hyperparameters optimization based on hyperopt library

    :param chain: chain to optimize
    :param task: task (classification, regression, ts_forecasting, clustering)
    :param iterations: max number of iterations
    """

    def __init__(self, chain, task, iterations=100,
                 max_lead_time: timedelta = timedelta(minutes=5),
                 log: Log = None):
        self.chain = chain
        self.task = task
        self.iterations = iterations
        self.max_seconds = int(max_lead_time.seconds)
        self.init_chain = None
        self.init_metric = None
        self.is_need_to_maximize = None

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    @abstractmethod
    def tune_chain(self, input_data, loss_function, loss_params=None):
        """
        Function for hyperparameters tuning on the chain

        :param input_data: data used for hyperparameter searching
        :param loss_function: function to minimize (or maximize) the metric,
        such function should take vector with true values as first values and
        predicted array as the second
        :param loss_params: dictionary with parameters for loss function
        :return fitted_chain: chain with optimized hyperparameters
        """
        raise NotImplementedError()

    def get_metric_value(self, data, chain, loss_function, loss_params):
        """
        Method calculates metric for algorithm validation

        :param data: InputData for validation
        :param chain: Chain to validate
        :param loss_function: function to minimize (or maximize)
        :param loss_params: parameters for loss function

        :return : value of loss function
        """

        # Make prediction
        if data.task.task_type == TaskTypesEnum.classification:
            test_target, preds = fit_predict_one_fold(data, chain)
        elif data.task.task_type == TaskTypesEnum.ts_forecasting:
            # For time series forecasting task in-sample forecasting is provided
            test_target, preds = ts_cross_validation(data, chain, log=self.log)
        else:
            test_target, preds = fit_predict_one_fold(data, chain)
            # Convert predictions into one dimensional array
            preds = np.ravel(np.array(preds))
            test_target = np.ravel(test_target)

        # Calculate metric
        if loss_params is None:
            metric = loss_function(test_target, preds)
        else:
            metric = loss_function(test_target, preds, **loss_params)
        return metric

    def init_check(self, data, loss_function, loss_params) -> None:
        """
        Method get metric on validation set before start optimization

        :param data: InputData for validation
        :param loss_function: function to minimize (or maximize)
        :param loss_params: parameters for loss function
        """
        self.log.info('Hyperparameters optimization start')

        # Train chain
        self.init_chain = deepcopy(self.chain)

        self.init_metric = self.get_metric_value(data=data,
                                                 chain=self.init_chain,
                                                 loss_function=loss_function,
                                                 loss_params=loss_params)

    def final_check(self, data, tuned_chain, loss_function, loss_params):
        """
        Method propose final quality check after optimization process

        :param data: InputData for validation
        :param tuned_chain: tuned chain
        :param loss_function: function to minimize (or maximize)
        :param loss_params: parameters for loss function
        """

        obtained_metric = self.get_metric_value(data=data,
                                                chain=tuned_chain,
                                                loss_function=loss_function,
                                                loss_params=loss_params)

        self.log.info('Hyperparameters optimization finished')

        prefix_tuned_phrase = 'Return tuned chain due to the fact that obtained metric'
        prefix_init_phrase = 'Return init chain due to the fact that obtained metric'

        # 5% deviation is acceptable
        deviation = (self.init_metric / 100.0) * 5

        if self.is_need_to_maximize is True:
            # Maximization
            init_metric = self.init_metric - deviation
            if obtained_metric >= init_metric:
                self.log.info(f'{prefix_tuned_phrase} {obtained_metric:.3f} equal or '
                              f'bigger than initial (- 5% deviation) {init_metric:.3f}')
                return tuned_chain
            else:
                self.log.info(f'{prefix_init_phrase} {obtained_metric:.3f} '
                              f'smaller than initial (- 5% deviation) {init_metric:.3f}')
                return self.init_chain
        else:
            # Minimization
            init_metric = self.init_metric + deviation
            if obtained_metric <= init_metric:
                self.log.info(f'{prefix_tuned_phrase} {obtained_metric:.3f} equal or '
                              f'smaller than initial (+ 5% deviation) {init_metric:.3f}')
                return tuned_chain
            else:
                self.log.info(f'{prefix_init_phrase} {obtained_metric:.3f} '
                              f'bigger than initial (+ 5% deviation) {init_metric:.3f}')
                return self.init_chain


def _greater_is_better(target, loss_function, loss_params) -> bool:
    """ Function checks is metric (loss function) need to be minimized or
    maximized

    :param target: target for define what problem is solving (max or min)
    :param loss_function: loss function
    :param loss_params: parameters for loss function

    :return : bool value is it good to maximize metric or not
    """

    if loss_params is None:
        metric = loss_function(target, target)
    else:
        try:
            metric = loss_function(target, target, **loss_params)
        except Exception:
            # Multiclass classification task
            metric = 1
    if int(round(metric)) == 0:
        return False
    else:
        return True
