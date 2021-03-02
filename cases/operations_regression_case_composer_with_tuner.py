import os
import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from matplotlib import pyplot as plt
import timeit

from pylab import rcParams
rcParams['figure.figsize'] = 18, 7
import warnings
warnings.filterwarnings('ignore')
import datetime

from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum

from sklearn.model_selection import train_test_split

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.chains.chain_tune import Tune

np.random.seed(10)


def run_experiment(file_path, chain, file_to_save):
    df = pd.read_csv(file_path)
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        np.array(df[['level_station_1', 'mean_temp', 'month', 'precip']]),
        np.array(df['level_station_2']),
        test_size=0.2,
        shuffle=True,
        random_state=10)

    obt_chains = []
    depths = []
    maes = []
    maes_first = []
    for i in range(0, 10):
        print(f'Iteration {i}')

        # Define regression task
        task = Task(TaskTypesEnum.regression)

        # Prepare data to train the model
        train_input = InputData(idx=np.arange(0, len(x_data_train)),
                                features=x_data_train,
                                target=y_data_train,
                                task=task,
                                data_type=DataTypesEnum.table)

        predict_input = InputData(idx=np.arange(0, len(x_data_test)),
                                  features=x_data_test,
                                  target=None,
                                  task=task,
                                  data_type=DataTypesEnum.table)

        available_model_types_secondary = ['ridge', 'lasso', 'dtreg',
                                           'xgbreg', 'adareg', 'knnreg',
                                           'linear', 'svr', 'poly_features', 'scaling',
                                           'ransac_lin_reg', 'rfe_lin_reg', 'pca',
                                           'ransac_non_lin_reg', 'rfe_non_lin_reg',
                                           'normalization']

        composer_requirements = GPComposerRequirements(
            primary=['one_hot_encoding'],
            secondary=available_model_types_secondary, max_arity=3,
            max_depth=8, pop_size=10, num_of_generations=3,
            crossover_prob=0.8, mutation_prob=0.8,
            max_lead_time=datetime.timedelta(minutes=5),
            add_single_operation_chains=True)

        metric_function = MetricsRepository().metric_by_id(
            RegressionMetricsEnum.MAE)
        builder = GPComposerBuilder(task=task).with_requirements(
            composer_requirements).with_metrics(metric_function).with_initial_chain(
            chain)
        composer = builder.build()

        obtained_chain = composer.compose_chain(data=train_input, is_visualise=False)

        print('Obtained chain')
        obtained_models = []
        for node in obtained_chain.nodes:
            print(str(node))
            obtained_models.append(str(node))
        depth = int(obtained_chain.depth)
        print(f'Chain depth {depth}')

        # Predict
        predicted_values = obtained_chain.predict(predict_input)
        preds = predicted_values.predict
        y_data_test = np.ravel(y_data_test)
        first_mae = mean_absolute_error(y_data_test, preds)
        print(f'MAE before tuning - {first_mae:.2f}')

        obtained_chain.fine_tune_all_nodes(train_input,
                                           max_lead_time=datetime.timedelta(minutes=2),
                                           iterations=30)
        # Fit it
        obtained_chain.fit_from_scratch(train_input)

        # Predict
        predicted_values = obtained_chain.predict(predict_input)
        preds = predicted_values.predict

        y_data_test = np.ravel(y_data_test)
        mae = mean_absolute_error(y_data_test, preds)

        print(f'RMSE - {mean_squared_error(y_data_test, preds, squared=False):.2f}')
        print(f'MAE - {mae:.2f}\n')

        obt_chains.append(obtained_models)
        maes.append(mae)
        depths.append(depth)

    report = pd.DataFrame({'Chain': obt_chains,
                           'Depth': depths,
                           'MAE before tuning': maes_first,
                           'MAE after tuning': maes})
    report.to_csv(file_to_save, index=False)


if __name__ == '__main__':

    node_encoder = PrimaryNode('one_hot_encoding')
    node_rans = SecondaryNode('ransac_lin_reg', nodes_from=[node_encoder])
    node_scaling = SecondaryNode('scaling', nodes_from=[node_rans])
    node_final = SecondaryNode('linear', nodes_from=[node_scaling])
    chain = Chain(node_final)

    run_experiment('../cases/data/river_levels/station_levels.csv', chain,
                   file_to_save='data/river_levels/old_composer_new_preprocessing_report_tuner.csv')








