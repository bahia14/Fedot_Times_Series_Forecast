import datetime
import os
import random

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from fedot.core.visualisation.opt_viz import PipelineEvolutionVisualiser

random.seed(1)
np.random.seed(1)


def calculate_validation_metric(pipeline: Pipeline, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = pipeline.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


def run_credit_scoring_problem(train_file_path, test_file_path,
                               timeout: datetime.timedelta = datetime.timedelta(minutes=5),
                               is_visualise=False,
                               with_tuning=False,
                               cache_path=None):
    task = Task(TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)
    dataset_to_validate = InputData.from_csv(test_file_path, task=task)

    # the search of the models provided by the framework that can be used as nodes in a pipeline for the selected task
    available_model_types = get_operations_for_task(task=task, mode='models')

    # the choice of the metric for the pipeline quality assessment during composition
    metric_function = ClassificationMetricsEnum.ROCAUC_penalty
    # the choice and initialisation of the GP search
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=3,
        max_depth=3, pop_size=20, num_of_generations=20,
        crossover_prob=0.8, mutation_prob=0.8, timeout=timeout)

    # GP optimiser parameters choice
    scheme_type = GeneticSchemeTypesEnum.parameter_free
    optimiser_parameters = GPGraphOptimiserParameters(genetic_scheme_type=scheme_type)

    # Create builder for composer and set composer params
    logger = default_log('FEDOT logger', verbose_level=4)
    builder = GPComposerBuilder(task=task).with_requirements(composer_requirements). \
        with_metrics(metric_function).with_optimiser_parameters(optimiser_parameters).with_logger(logger=logger)

    if cache_path:
        builder = builder.with_cache(cache_path)

    # Create GP-based composer
    composer = builder.build()

    # the optimal pipeline generation by composition - the most time-consuming task
    pipeline_evo_composed = composer.compose_pipeline(data=dataset_to_compose,
                                                      is_visualise=True)

    if with_tuning:
        pipeline_evo_composed.fine_tune_all_nodes(loss_function=roc_auc,
                                                  loss_params=None,
                                                  input_data=dataset_to_compose,
                                                  iterations=20)

    pipeline_evo_composed.fit(input_data=dataset_to_compose)

    composer.history.write_composer_history_to_csv()

    if is_visualise:
        visualiser = PipelineEvolutionVisualiser()

        composer.log.debug('History visualization started')
        visualiser.visualise_history(composer.history)
        composer.log.debug('History visualization finished')

        composer.log.debug('Best pipeline visualization started')
        pipeline_evo_composed.show()
        composer.log.debug('Best pipeline visualization finished')

    # the quality assessment for the obtained composite models
    roc_on_valid_evo_composed = calculate_validation_metric(pipeline_evo_composed,
                                                            dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')

    return roc_on_valid_evo_composed


def get_scoring_data():
    # the dataset was obtained from https://www.kaggle.com/c/GiveMeSomeCredit

    # a dataset that will be used as a train and test set during composition

    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    return full_path_train, full_path_test


if __name__ == '__main__':
    full_path_train, full_path_test = get_scoring_data()
    run_credit_scoring_problem(full_path_train,
                               full_path_test,
                               is_visualise=True,
                               with_tuning=True,
                               cache_path='credit_scoring_problem_cache')
