from fedot.api.main import Fedot
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode
from fedot.core.models.atomized_model import AtomizedModel
from fedot.core.utils import project_root


def run_additional_learning_example():
    train_data_path = f'{project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    auto_model = Fedot(problem=problem, seed=42, preset='light',
                       composer_params={'initial_chain': Chain(PrimaryNode('logit '))})
    auto_model.fit(features=train_data_path, target='target')
    auto_model.predict_proba(features=test_data_path)
    print('auto_model', auto_model.get_metrics())

    atomized_model = Chain(PrimaryNode(model_type=AtomizedModel(auto_model.current_model)))
    non_atomized_model = auto_model.current_model

    auto_model_from_atomized = Fedot(problem=problem, seed=42, preset='light',
                                     composer_params={'initial_chain': atomized_model})
    auto_model_from_atomized.fit(features=train_data_path, target='target')
    auto_model_from_atomized.predict_proba(features=test_data_path)
    print('auto_model_from_atomized', auto_model_from_atomized.get_metrics())

    auto_model_from_chain = Fedot(problem=problem, seed=42, preset='light',
                                  composer_params={'initial_chain': non_atomized_model})
    auto_model_from_chain.fit(features=train_data_path, target='target')
    auto_model_from_chain.predict_proba(features=test_data_path)
    print('auto_model_from_chain', auto_model_from_chain.get_metrics())


if __name__ == '__main__':
    run_additional_learning_example()
