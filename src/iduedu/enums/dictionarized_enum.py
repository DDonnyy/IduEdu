from typing import Any


class DictionarizedEnum:
    def __init__(self, data: dict):
        self._data: dict = data
    
    def get(self, key: str, default: Any):
        if key in self._data:
            return self._data[key]
        return default
    
    def set(self, key: str, value: Any):
        if key not in self._data:
            raise AttributeError(f"No such key: {key}")
        if type(self._data[key]) is not type(value):
            raise AttributeError(f"Value type isn't equal to initial field type")
        self._data[key] = value
    
    def add(self, key: str, value: Any):
        if key in self._data:
            raise AttributeError(f"Key already exists: {key}")
        self._data[key] = value
