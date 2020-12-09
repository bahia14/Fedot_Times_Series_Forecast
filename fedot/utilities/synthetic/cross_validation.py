import time
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed

from fedot.core.models.data import InputData
from fedot.utilities.synthetic.split import KFold, indexable


def cross_validate(chain: 'Chain', data: 'InputData', *, scoring=None, cv=None,
                   n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                   return_train_score=False, return_estimator=False) -> dict:
    """
    Evaluate metric(s) by cross-validation and also record fit/score times.

    :param chain: the chain to use to fit the data.
    :param data: the data to fit has type of instance: 'InputData'.
    #TODO what the type of scorer to pass
    :param scoring: the scorer ???
    :param cv: cross-validation generator or an iterable (int or instance of BaseCrossValidator).
    :param n_jobs: the number of CPUs to use to do the computation.
    :param verbose: the verbosity level.
    :param fit_params: parameters to pass to the fit method of the estimator.
    :param pre_dispatch: controls the number of jobs that get dispatched during parallel execution.
    :param return_train_score: whether to include train scores.
    :param return_estimator: whether to return the estimators fitted on each split.

    :return train_scores: float score on training set, returned only if `return_train_score` is `True`.
    :return test_scores: float score on testing set.
    :return fit_time: time in float spent for fitting in seconds.
    :return score_time: time in float spent for scoring in seconds.
    :return estimator: the fitted estimator object, returned only if `return_estimator` is `True`.
    """

    features, target = indexable(data.features, data.target)

    cv = 5 if cv is None else cv
    if isinstance(cv, int):
        cv = KFold(cv)

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)

    # We clone the estimator to make sure that all the folds are independent.
    scores = parallel(
        delayed(_fit_and_score)(
            deepcopy(chain), data, features, target, scoring, train, test, verbose,
            fit_params, return_train_score=return_train_score, return_estimator=return_estimator)
        for train, test in cv.split(features, target))

    zipped_scores = list(zip(*scores))
    if return_train_score:
        train_scores = zipped_scores.pop(0)
    if return_estimator:
        fitted_estimators = zipped_scores.pop()
    test_scores, fit_times, score_times = zipped_scores

    ret = {'fit_time': np.array(fit_times), 'score_time': np.array(score_times)}

    if return_estimator:
        ret['estimator'] = fitted_estimators

    ret['test'] = np.array(test_scores)
    if return_train_score:
        ret['train'] = np.array(train_scores)

    return ret


def _fit_and_score(estimator, data, features, target, scorer, train_indexes, test_indexes, verbose,
                   fit_params, return_train_score=False, return_estimator=False) -> list:
    """Fit estimator and compute scores for a given dataset split."""

    msg = ''

    if verbose > 1:
        if fit_params is None:
            msg = ''
        else:
            msg = f"{(', '.join(f'{k}={v}' for k, v in fit_params.items()))}"
        print(f"[CrossVal] {msg} {(64 - len(msg)) * '.'}")

    fit_params = fit_params if fit_params is not None else {}
    train_scores = {}
    start_time = time.time()

    X_train = [features[idx] for idx in train_indexes]
    y_train = [target[idx] for idx in train_indexes]
    X_test = [features[idx] for idx in test_indexes]
    y_test = [target[idx] for idx in test_indexes]

    train = InputData(features=X_train, target=y_train,
                      idx=np.arange(0, len(X_train)),
                      task=data.task, data_type=data.data_type)

    test = InputData(features=X_test, target=y_test,
                     idx=np.arange(0, len(X_test)),
                     task=data.task, data_type=data.data_type)

    try:
        estimator.fit(train, **fit_params)
    except Exception as ex:
        raise
    else:
        fit_time = time.time() - start_time
        test_scores = scorer(estimator, test)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = scorer(estimator, train)

    if verbose > 2:
        if isinstance(test_scores, dict):
            for scorer_name in sorted(test_scores):
                msg += ", %s=" % scorer_name
                if return_train_score:
                    msg += f"(train={train_scores[scorer_name]:.3f},"
                    msg += f" test={test_scores[scorer_name]:.3f})"
                else:
                    msg += f"{test_scores[scorer_name]:.3f}"
        else:
            msg += ", score="
            msg += (f"{test_scores:.3f}" if not return_train_score else
                    f"(train={train_scores:.3f}, test={test_scores:.3f})")

    if verbose > 1:
        total_time = score_time + fit_time
        print(_message_with_time(msg, total_time))

    ret = [train_scores, fit_time, score_time, test_scores] if return_train_score \
        else [fit_time, score_time, test_scores]

    if return_estimator:
        ret.append(estimator)
    return ret


def _message_with_time(message: str, time_spent: float) -> str:
    """Create one line message for logging purposes.

    :param message: string message.
    :param time_spent: int time in seconds.
    :return: string.
    """

    start_message = f"[CrossVal] "

    if time_spent > 60:
        time_str = f"{(time_spent / 60):.1f}min"
    else:
        time_str = f" {time_spent:.1f}s"
    end_message = f" {message}, total={time_str}"
    dots_len = (70 - len(start_message) - len(end_message))
    return f"{start_message, dots_len * '.', end_message}"
