from abc import ABCMeta, abstractmethod, ABC
import numpy as np
from itertools import combinations


class BaseCrossValidator(metaclass=ABCMeta):
    """Base class for all cross-validators"""

    def split(self, features, target=None):
        """Generate indices to split data into training and test set.

        :param features: the training data.
        :param target: the target variable.
        :return yields:
            train: ndarray The training set indices for that split.
            test: ndarray The testing set indices for that split.
        """

        features, target = indexable(features, target)
        indices = np.arange(len(features))
        for test_index in self._iter_test_masks(features, target):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_masks(self, features=None, target=None):
        """Generates boolean masks corresponding to test sets."""

        for test_index in self._iter_test_indices(features, target):
            test_mask = np.zeros(len(features), dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    @abstractmethod
    def _iter_test_indices(self, features, target=None):
        """Generates integer indices corresponding to test sets."""

        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, features, target=None):
        """Returns the number of splitting iterations in the cross-validator"""

        raise NotImplementedError


class LeaveOneOut(BaseCrossValidator):
    """Leave-One-Out cross-validator

    Provides train/test indices to split data in train/test sets. Each
    sample is used once as a test set (singleton) while the remaining
    samples form the training set.

    Due to the high number of test sets (which is the same as the
    number of samples) this cross-validation method can be very costly.
    """

    def _iter_test_indices(self, features, target=None):
        n_samples = len(features)
        if n_samples <= 1:
            raise ValueError(
                'Cannot perform LeaveOneOut with n_samples={}.'.format(
                    n_samples)
            )
        return range(n_samples)

    def get_n_splits(self, features, target=None):
        """Returns the number of splitting iterations in the cross-validator"""

        if features is None:
            raise ValueError("The 'X' parameter should not be None.")
        return len(features)


class LeavePOut(BaseCrossValidator):
    """Leave-P-Out cross-validator

    Provides train/test indices to split data in train/test sets. This results
    in testing on all distinct samples of size p, while the remaining n - p
    samples form the training set in each iteration.

    Due to the high number of iterations which grows combinatorically with the
    number of samples this cross-validation method can be very costly.
    """

    def __init__(self, p):
        super().__init__()
        self.p = p

    def _iter_test_indices(self, features, target=None):
        n_samples = len(features)
        if n_samples <= self.p:
            raise ValueError(
                'p={} must be strictly less than the number of '
                'samples={}'.format(self.p, n_samples)
            )
        for combination in combinations(range(n_samples), self.p):
            yield np.array(combination)

    def get_n_splits(self, features, target=None):
        """Returns the number of splitting iterations in the cross-validator"""

        if features is None:
            raise ValueError("The 'X' parameter should not be None.")
        return int(comb(len(features), self.p, exact=True))


class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Base class for KFold and StratifiedKFold"""

    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state):
        super().__init__()
        if not isinstance(n_splits, int):
            raise ValueError(f"The number of folds must be of Int type. "
                             f"{n_splits} of type {type(n_splits)} was passed.")
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                f"k-fold cross-validation requires at least one"
                f" train/test split by setting n_splits=2 or more,"
                f" got n_splits={n_splits}.")

        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle must be True or False;"
                            f" got {shuffle}")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, features, target=None):
        """Generate indices to split data into training and test set.

        :param features: the training data.
        :param target: the target variable.
        :return yields:
            train: ndarray The training set indices for that split.
            test: ndarray The testing set indices for that split.
        """

        features, target = indexable(features, target)
        n_samples = len(features)
        if self.n_splits > n_samples:
            raise ValueError(
                (f"Cannot have number of splits n_splits={self.n_splits} greater"
                 f" than the number of samples: n_samples={n_samples}."))

        for train, test in super().split(features, target):
            yield train, test

    def get_n_splits(self, features=None, target=None):
        """Returns the number of splitting iterations in the cross-validator"""

        return self.n_splits


class KFold(_BaseKFold):
    """K-Folds cross-validator

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the k - 1 remaining folds form the training set.

    :param n_splits: int, default=5
    :prams shuffle: bool, default=False
    :param random_state: int, default=None
    """

    def __init__(self, n_splits=5, *, shuffle=False,
                 random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def _iter_test_indices(self, features, target=None):
        n_samples = len(features)
        indices = np.arange(n_samples)
        if self.shuffle:
            self.random_state.shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop


class StratifiedKFold(_BaseKFold, ABC):
    """Stratified K-Folds cross-validator

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of KFold that returns stratified folds.
    The folds are made by preserving the percentage of samples for each class.

    :param n_splits: int, default=5
    :param shuffle: bool, default=False
    :param random_state: int, default=None
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def _make_test_folds(self, features, target=None):
        target = np.asarray(target)
        target = np.ravel(target)

        _, y_idx, y_inv = np.unique(target, return_index=True, return_inverse=True)
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        if all(self.n_splits > y_counts):
            raise ValueError(f"n_splits={self.n_splits} cannot be greater than the"
                             f" number of members in each class.")

        # Determine the optimal number of samples from each class in each fold,
        # using round robin over the sorted y.
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [np.bincount(y_order[i::self.n_splits], minlength=n_classes)
             for i in range(self.n_splits)])

        test_folds = np.empty(len(target), dtype='i')
        for k in range(n_classes):
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                self.random_state.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

    def _iter_test_masks(self, features=None, target=None):
        test_folds = self._make_test_folds(features, target)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, features, target=None):
        """Generate indices to split data into training and test set."""

        return super().split(features, target)


class TimeSeriesSplit(_BaseKFold, ABC):
    """Time Series cross-validator

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before,
    and thus shuffling in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.

    :param n_splits: int, default=5
        Number of splits. Must be at least 2.

    :param max_train_size: int, default=None
        Maximum size for a single training set.
    """

    def __init__(self, n_splits=5, *, max_train_size=None):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, features, target=None):
        """Generate indices to split data into training and test set."""

        features, target = indexable(features, target)
        n_samples = len(features)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        if n_folds > n_samples:
            raise ValueError(
                (f"Cannot have number of folds ={n_folds} greater"
                 f" than the number of samples: {n_samples}."))
        indices = np.arange(n_samples)
        test_size = (n_samples // n_folds)
        test_starts = range(test_size + n_samples % n_folds,
                            n_samples, test_size)
        for test_start in test_starts:
            if self.max_train_size and self.max_train_size < test_start:
                yield (indices[test_start - self.max_train_size:test_start],
                       indices[test_start:test_start + test_size])
            else:
                yield (indices[:test_start],
                       indices[test_start:test_start + test_size])


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions and
    whether all objects in arrays have the same shape or length.

    :params *arrays: list or tuple of input objects.
    """

    lengths = [len(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(f"Found input variables with inconsistent numbers of"
                         f" samples: {[int(l) for l in lengths]}")


def _make_indexable(iterable):
    """Ensure iterable supports indexing or convert to an indexable variant.

    :param iterable: list | dataframe | array | None
    """

    if hasattr(iterable, "__getitem__") or hasattr(iterable, "iloc"):
        return iterable
    elif iterable is None:
        return iterable
    return np.array(iterable)


def indexable(*iterables) -> list:
    """Make arrays indexable for cross-validation.

    :params *iterables: lists | dataframes | arrays
    """

    result = [_make_indexable(X) for X in iterables]
    check_consistent_length(*result)
    return result


def comb(n, k, exact=False, repetition=False):
    """The number of combinations of N things taken k at a time.

    :param n: total number.
    :param k: to take objects from n.
    :param exact: if `exact` is False, then floating point precision is used, otherwise exact long integer is computed.
    :param repetition: if `repetition` is True, then the number of combinations with repetition is computed.
    :return int, float: the total number of combinations.
    """

    if (k <= n) & (n >= 0) & (k >= 0):
        if repetition:
            return comb(n + k - 1, k, exact)
        else:
            return sum(1 for _ in comb(n, k, exact))
    else:
        return 0
