from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param

class TransformerProxy(Transformer):

    def __init__(self):
        super(TransformerProxy, self).__init__()
        self.transformer = Param(self, "transformer", "")

    def set_transformer(self, transformer):
        self._paramMap[self.transformer] = transformer
        return self

    def get_transformer(self):
        return self.getOrDefault(self.transformer)

    def _transform(self, dataset):
        return self.get_transformer().transform(dataset)

