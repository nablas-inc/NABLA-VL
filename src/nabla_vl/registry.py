from typing import Any, List


class Registry(object):
    def __init__(self) -> None:
        self.str_to_cls = {}

    def __getitem__(self, name: str) -> Any:
        return self.str_to_cls[name]

    def list_registered_modules(self) -> List[str]:
        return list(self.str_to_cls.keys())

    def register_cls(self) -> Any:
        def fn(cls):
            name = cls.__name__
            self.str_to_cls[name] = cls
            return cls

        return lambda cls: fn(cls)


DATASETS = Registry()
