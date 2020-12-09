from cases.data.data_utils import get_scoring_case_data_paths
from test.test_chain_import_export import create_chain
from fedot.core.models.data import InputData
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.utilities.synthetic.cross_validation import cross_validate
from examples.utils import create_multi_clf_examples_from_excel


def test_cross_val_regression_correct():
    cv = 2
    chain = create_chain()
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    scoring = ClassificationMetricsEnum.f1
    scorer = MetricsRepository().metric_by_id(scoring)

    result = cross_validate(chain, train_data, scoring=scorer, cv=cv,
                            return_estimator=True, return_train_score=True,
                            verbose=3)

    actual_score = result['train']

    chain = create_chain()
    chain.fit(train_data)
    expected_score = scorer(chain, train_data)

    assert len(actual_score) == cv
    assert actual_score.mean() < expected_score


def test_cross_val_correct():
    cv = 5
    chain = create_chain()
    file_path_first = r'./data/example1.xlsx'

    train_file_path, test_file_path = create_multi_clf_examples_from_excel(file_path_first)
    train_data = InputData.from_csv(train_file_path)

    scoring = ClassificationMetricsEnum.f1
    scorer = MetricsRepository().metric_by_id(scoring)

    result = cross_validate(chain, train_data, scoring=scorer, cv=cv,
                            return_estimator=True, return_train_score=True,
                            verbose=3)

    actual_score = result['train']

    chain = create_chain()
    chain.fit(train_data)
    expected_score = scorer(chain, train_data)

    assert len(actual_score) == cv
    assert actual_score.mean() < expected_score


test_cross_val_correct()
