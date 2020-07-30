class FittedModelCache:
    def __init__(self, related_node: 'Node'):
        self._local_cached_models = {}
        self._related_node_ref = related_node

    def append(self, fitted_model):
        self._local_cached_models[self._related_node_ref.descriptive_id] = fitted_model

    def import_from_other_cache(self, other_cache: 'FittedModelCache'):
        for entry_key in other_cache._local_cached_models.keys():
            self._local_cached_models[entry_key] = other_cache._local_cached_models[entry_key]

    def clear(self):
        self._local_cached_models = {}

    @property
    def actual_cached_state(self):
        found_model = self._local_cached_models.get(self._related_node_ref.descriptive_id, None)
        return found_model


class SharedCache(FittedModelCache):
    def __init__(self, related_node: 'Node', global_cached_models: dict):
        super().__init__(related_node)
        self._global_cached_models = global_cached_models

    def append(self, fitted_model):
        super().append(fitted_model)
        if self._global_cached_models is not None:
            self._global_cached_models[self._related_node_ref.descriptive_id] = fitted_model

    @property
    def actual_cached_state(self):
        found_model = super().actual_cached_state

        if not found_model and self._global_cached_models:
            found_model = self._global_cached_models.get(self._related_node_ref.descriptive_id, None)
        return found_model
