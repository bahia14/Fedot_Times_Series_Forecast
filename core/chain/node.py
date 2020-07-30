from abc import ABC, abstractmethod
from collections import namedtuple
from copy import copy
from datetime import timedelta
from typing import Callable, List, Optional

from core.chain.cache import FittedModelCache
from core.data.data import InputData, OutputData
from core.data.preprocessing import preprocessing_func_for_data
from core.data.transformation import transformation_function_for_data
from core.models.model import Model

CachedState = namedtuple('CachedState', 'preprocessor model')


class Node(ABC):
    def __init__(self, nodes_from: Optional[List['Node']],
                 model_type: str):
        self.nodes_from = nodes_from
        self.model = Model(id=model_type)
        self.cache = FittedModelCache(self)
        self.manual_preprocessing_func = None

    @property
    def descriptive_id(self):
        return self._descriptive_id_recursive(visited_nodes=[])

    def _descriptive_id_recursive(self, visited_nodes):
        node_label = self.model.description
        if self.manual_preprocessing_func:
            node_label = f'{node_label}_custom_preprocessing={self.manual_preprocessing_func.__name__}'
        full_path = ''
        if self in visited_nodes:
            return 'ID_CYCLED'
        visited_nodes.append(self)
        if self.nodes_from:
            previous_items = []
            for parent_node in self.nodes_from:
                previous_items.append(f'{parent_node._descriptive_id_recursive(copy(visited_nodes))};')
            previous_items.sort()
            previous_items_str = ';'.join(previous_items)

            full_path += f'({previous_items_str})'
        full_path += f'/{node_label}'
        return full_path

    @property
    def model_tags(self) -> List[str]:
        return self.model.metadata.tags

    def output_from_prediction(self, input_data, prediction):
        return OutputData(idx=input_data.idx,
                          features=input_data.features,
                          predict=prediction, task=input_data.task,
                          data_type=self.model.output_datatype(input_data.data_type),
                          target=input_data.target)

    def _transform(self, input_data: InputData):
        transformed_data = transformation_function_for_data(
            input_data_type=input_data.data_type,
            required_data_types=self.model.metadata.input_types)(input_data)
        return transformed_data

    def _preprocess(self, data: InputData):
        preprocessing_func = preprocessing_func_for_data(data, self)

        if not self.cache.actual_cached_state:
            # if fitted preprocessor not found in cache
            preprocessing_strategy = \
                preprocessing_func().fit(data.features)
        else:
            # if fitted preprocessor already exists
            preprocessing_strategy = self.cache.actual_cached_state.preprocessor

        data.features = preprocessing_strategy.apply(data.features)

        return data, preprocessing_strategy

    @abstractmethod
    def fit(self, verbose: bool = False):
        pass

    @abstractmethod
    def predict(self, verbose: bool = False):
        pass

    @abstractmethod
    def fine_tune(self, max_lead_time: timedelta = timedelta(minutes=5), iterations: int = 30):
        pass

    def fit_with_data(self, input_data: InputData, verbose=False) -> OutputData:
        transformed = self._transform(input_data)
        preprocessed_data, preproc_strategy = self._preprocess(transformed)

        if not self.cache.actual_cached_state:
            if verbose:
                print('Cache is not actual')

            cached_model, model_predict = self.model.fit(data=preprocessed_data)
            self.cache.append(CachedState(preprocessor=copy(preproc_strategy),
                                          model=cached_model))
        else:
            if verbose:
                print('Model were obtained from cache')

            model_predict = self.model.predict(fitted_model=self.cache.actual_cached_state.model,
                                               data=preprocessed_data)

        return self.output_from_prediction(input_data, model_predict)

    def predict_with_data(self, input_data: InputData, verbose=False) -> OutputData:
        transformed = self._transform(input_data)
        preprocessed_data, _ = self._preprocess(transformed)

        if not self.cache:
            raise ValueError('Model must be fitted before predict')

        model_predict = self.model.predict(fitted_model=self.cache.actual_cached_state.model,
                                           data=preprocessed_data)

        return self.output_from_prediction(input_data, model_predict)

    def fine_tune_with_data(self, input_data: InputData,
                            max_lead_time: timedelta = timedelta(minutes=5), iterations: int = 30):

        transformed = self._transform(input_data)
        preprocessed_data, preproc_strategy = self._preprocess(transformed)

        fitted_model, _ = self.model.fine_tune(preprocessed_data,
                                               max_lead_time=max_lead_time,
                                               iterations=iterations)

        self.cache.append(CachedState(preprocessor=copy(preproc_strategy),
                                      model=fitted_model))

    def __str__(self):
        model = f'{self.model}'
        return model

    @property
    def ordered_subnodes_hierarchy(self) -> List['Node']:
        nodes = [self]
        if self.nodes_from:
            for parent in self.nodes_from:
                nodes += parent.ordered_subnodes_hierarchy
        return nodes

    @property
    def custom_params(self) -> dict:
        return self.model.params

    @custom_params.setter
    def custom_params(self, params):
        self.model.params = params


class DataNode(Node):
    def __init__(self, model_type: Optional[str] = None):
        if not model_type:
            model_type = 'data_source'
        super().__init__(nodes_from=None, model_type=model_type)
    def fit(self, verbose: bool = False):
        data = self.cache.actual_cached_state.model
        return self.output_from_prediction(input_data=data, prediction=data.features)

    def predict(self, verbose: bool = False):
        data = self.cache.actual_cached_state.model
        raise self.output_from_prediction(input_data=data, prediction=data.features)

    def fine_tune(self, max_lead_time: timedelta = timedelta(minutes=5),
                  iterations: int = 30):
        raise NotImplementedError()


class ModelNode(Node):
    def __init__(self, model_type: str, nodes_from: Optional[List['Node']] = None,
                 manual_preprocessing_func: Optional[Callable] = None,
                 local_target=None):
        nodes_from = [] if nodes_from is None else nodes_from
        super().__init__(nodes_from=nodes_from, model_type=model_type)
        self.manual_preprocessing_func = manual_preprocessing_func
        self.local_target = local_target

    def fit(self, verbose=False) -> OutputData:
        if verbose:
            print(f'Trying to fit node with model: {self.model}')

        secondary_input = self._input_from_parents(parent_operation='fit',
                                                   verbose=verbose)

        return super().fit_with_data(input_data=secondary_input)

    def predict(self, verbose=False) -> OutputData:
        if verbose:
            print(f'Obtain prediction in node with model: {self.model}')

        secondary_input = self._input_from_parents(parent_operation='predict',
                                                   verbose=verbose)

        return super().predict_with_data(input_data=secondary_input)

    def fine_tune(self,
                  max_lead_time: timedelta = timedelta(minutes=5), iterations: int = 30,
                  verbose: bool = False):
        if verbose:
            print(f'Tune all parent nodes in secondary node with model: {self.model}')

        secondary_input = self._input_from_parents(parent_operation='fine_tune',
                                                   max_tune_time=max_lead_time, verbose=verbose)

        return super().predict_with_data(input_data=secondary_input)

    def _nodes_from_with_fixed_order(self):
        if self.nodes_from is not None:
            return sorted(self.nodes_from, key=lambda node: node.descriptive_id)

    def _input_from_parents(self,
                            parent_operation: str,
                            max_tune_time: Optional[timedelta] = None,
                            verbose=False) -> InputData:
        if len(self.nodes_from) == 0:
            raise ValueError()

        if verbose:
            print(f'Fit all parent nodes in secondary node with model: {self.model}')

        parent_nodes = self._nodes_from_with_fixed_order()

        are_prev_nodes_affect_target = \
            ['affects_target' in parent_node.model_tags for parent_node in parent_nodes]
        if any(are_prev_nodes_affect_target):
            # is the previous model is the model that changes target
            parent_results, target = _combine_parents_that_affects_target(parent_nodes,
                                                                          parent_operation)
        else:
            parent_results, target = _combine_parents_simple(parent_nodes,
                                                             parent_operation,
                                                             max_tune_time)

        if self.local_target:
            target = self.local_target

        secondary_input = InputData.from_predictions(outputs=parent_results,
                                                     target=target)

        return secondary_input


def _combine_parents_that_affects_target(parent_nodes: List[Node],
                                         parent_operation: str):
    if len(parent_nodes) > 1:
        raise NotImplementedError()

    if parent_operation == 'predict':
        parent_result = parent_nodes[0].predict()
    elif parent_operation == 'fit' or parent_operation == 'fine_tune':
        parent_result = parent_nodes[0].fit()
    else:
        raise NotImplementedError()

    target = parent_result.predict
    return [parent_result], target


def _combine_parents_simple(parent_nodes: List[Node],
                            parent_operation: str,
                            max_tune_time: Optional[timedelta]):
    parent_results = []
    for parent in parent_nodes:
        if parent_operation == 'predict':
            parent_results.append(parent.predict())
        elif parent_operation == 'fit':
            parent_results.append(parent.fit())
        elif parent_operation == 'fine_tune':
            parent.fine_tune(max_lead_time=max_tune_time)
            parent_results.append(parent.predict())
        else:
            raise NotImplementedError()

    # TODO change to main data
    target = parent_results[0].target

    return parent_results, target
