import os
from typing import Tuple

from sklearn.metrics import mean_squared_error as mse

from core.chain.chain import Chain, DataNode, ModelNode
from core.data.data import InputData, OutputData
from core.data.preprocessing import SimpleStrategy
from core.repository.tasks import Task, TaskTypesEnum


def get_rmse_value(prediction: OutputData, data: InputData) -> (float, float):
    return mse(y_true=data.target, y_pred=prediction.predict, squared=False)


def get_sys_dyn_data():
    test_file_path = str(os.path.dirname(__file__))
    task = Task(TaskTypesEnum.regression)
    datas = []

    for mode in ['train', 'test']:
        for file_id in ['y1', 'y2', 'yfin']:
            file_path = os.path.join(test_file_path,
                                     f'data/system_dynamics/multi_target_{file_id}_{mode}.csv')
            data = InputData.from_csv(file_path, task=task)
            datas.append(data)

    return datas


def get_regr_coeffs(node: ModelNode) -> Tuple:
    model = node.cache.actual_cached_state.model
    coeffs = list(model.coef_)
    coeffs.append(model.intercept_)
    coeffs = [round(coeff, 2) for coeff in coeffs]
    return tuple(coeffs)


def test_system_dynamics_chain():
    # Model structure
    # Data1    Data2    Data3 + Node1
    #   +        +        +
    # Node1 -> Node2 -> Node3 -> final
    data_first_train, data_second_train, data_third_train,\
        data_first_test, data_second_test, data_third_test= get_sys_dyn_data()

    real_regression_coeffs = [(2.0, 3.0, 5.0), (6.0, -5.0, 9.0), (0.5, 3.0, -1.0, -5.0)]

    first_data_node = DataNode()
    second_data_node = DataNode()
    third_data_node = DataNode()

    # no scaling to preserve regression coefficients
    preprocessing = SimpleStrategy

    first_model_node = ModelNode(model_type='linear',
                                 nodes_from=[first_data_node],
                                 manual_preprocessing_func=preprocessing,
                                 local_target=data_first_train.target)

    second_model_node = ModelNode(model_type='linear',
                                  nodes_from=[first_model_node, second_data_node],
                                  manual_preprocessing_func=preprocessing,
                                  local_target=data_second_train.target)

    final_model_node = ModelNode(model_type='linear',
                                 nodes_from=[first_model_node, second_model_node, third_data_node],
                                 manual_preprocessing_func=preprocessing,
                                 local_target=data_third_train.target)

    chain = Chain(final_model_node)

    train_input_data_dict = {first_data_node: data_first_train,
                       second_data_node: data_second_train,
                       third_data_node: data_third_train}

    test_input_data_dict = {first_data_node: data_first_test,
                             second_data_node: data_second_test,
                             third_data_node: data_third_test}

    chain.fit(train_input_data_dict)

    assert get_regr_coeffs(first_model_node) == real_regression_coeffs[0]
    assert get_regr_coeffs(second_model_node) == real_regression_coeffs[1]
    assert get_regr_coeffs(final_model_node) == real_regression_coeffs[2]

    prediction_test = chain.predict(test_input_data_dict)

    error = get_rmse_value(prediction_test, data_third_test)
    assert error < 0.001
