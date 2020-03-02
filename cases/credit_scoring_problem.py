import os
import random

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.chain import Chain
from core.composer.composer import DummyChainTypeEnum
from core.composer.composer import DummyComposer
from core.composer.random_composer import RandomSearchComposer
from core.composer.visualisation import ChainVisualiser
from core.models.data import InputData
from core.models.model import MLP
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum
from core.utils import project_root

random.seed(1)
np.random.seed(1)


def calculate_validation_metric_for_scoring_model(chain: Chain, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = chain.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


# the dataset was obtained from https://www.kaggle.com/kashnitsky/a5-demo-logit-and-rf-for-credit-scoring

# a dataset that will be used as a train and test set during composition
file_path_train = 'cases/data/scoring/scoring_train.csv'
full_path_train = os.path.join(str(project_root()), file_path_train)
dataset_to_compose = InputData.from_csv(full_path_train)

# a dataset for a final validation of the composed model
file_path_test = 'cases/data/scoring/scoring_test.csv'
full_path_test = os.path.join(str(project_root()), file_path_test)
dataset_to_validate = InputData.from_csv(full_path_test)

# the search of the models provided by the framework that can be used as nodes in a chain for the selected task
models_repo = ModelTypesRepository()
available_model_names = models_repo.search_model_types_by_attributes(
    desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                           output_type=CategoricalDataTypesEnum.vector,
                                           task_type=MachineLearningTasksEnum.classification,
                                           can_be_initial=True,
                                           can_be_secondary=True))

models_impl = [models_repo.model_by_id(model_name) for model_name in available_model_names]

# the choice of the metric for the chain quality assessment during composition
metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

# the choice and initialisation of the dummy_composer
dummy_composer = DummyComposer(DummyChainTypeEnum.hierarchical_2lev)
# the choice and initialisation of the random_search
random_composer = RandomSearchComposer(iter_num=3)

# the optimal chain generation by composition - the most time-consuming task
chain_random_composed = random_composer.compose_chain(data=dataset_to_compose,
                                                      initial_chain=None,
                                                      primary_requirements=models_impl,
                                                      secondary_requirements=models_impl,
                                                      metrics=metric_function)

chain_static = dummy_composer.compose_chain(data=dataset_to_compose,
                                            initial_chain=None,
                                            primary_requirements=models_impl,
                                            secondary_requirements=models_impl,
                                            metrics=metric_function)

# the single-model variant of optimal chain
chain_single = DummyComposer(DummyChainTypeEnum.flat).compose_chain(data=dataset_to_compose,
                                                                    initial_chain=None,
                                                                    primary_requirements=[MLP()],
                                                                    secondary_requirements=[],
                                                                    metrics=metric_function)

print("Composition finished")

visualiser = ChainVisualiser()
visualiser.visualise(chain_static)
visualiser.visualise(chain_random_composed)

# the quality assessment for the obtained composite models
roc_on_valid_static = calculate_validation_metric_for_scoring_model(chain_static, dataset_to_validate)
roc_on_valid_single = calculate_validation_metric_for_scoring_model(chain_single, dataset_to_validate)
roc_on_valid_random_composed = calculate_validation_metric_for_scoring_model(chain_random_composed, dataset_to_validate)

print(f'Composed ROC AUC is {round(roc_on_valid_random_composed, 3)}')
print(f'Static ROC AUC is {round(roc_on_valid_static, 3)}')
print(f'Single-model ROC AUC is {round(roc_on_valid_single, 3)}')