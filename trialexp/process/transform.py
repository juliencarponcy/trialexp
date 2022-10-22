from abc import ABC, abstractmethod


class Transform(ABC):

    def __init__(self, params_dict):
        self.params = params_dict

    def check_params(self, params):
        ...
    
    def get_col_names():
        ...

    def get_params():
        ...


    @abstractmethod
    def transform(self, data, params):
        ...

class TransformFactory(ABC):

    @abstractmethod
    def create_transform() -> Transform:
        ...


class SessionTransformFactory(TransformFactory):
    def __init__(self, params_dict) -> None:
        super().__init__()

    def check_data(data):
        ...

    def prepare_data(data):
        ...

    def create_transform() -> Transform:
        ...


class TrialTransformFactory(TransformFactory):

    def check_data():
        ...

    def create_transform() -> Transform:
        ...

    
class FreqTransform(Transform):

    def transform(self, data, params):
        ...

    def get_freq(): 
        ...


class BoolTransform(Transform):

    def check_params(self, params):
        return super().check_params(params)

    def transform(self, data, params):
        ...



class DimTransform(Transform):

    def get_weights():
        ...
