import json
import statistics

from abc import ABC, abstractmethod

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import (
    HasInputCol, HasOutputCol, Param
)
from pyspark.sql.functions import udf
from pyspark.sql.types import (
    ArrayType, StringType
)

from external.modules.features import BaseFeatureTransformer

class FeatureFilterTransformer(BaseFeatureTransformer):
    
    def __init__(self, keep: list, **kwargs):
        super().__init__(kwargs)
        self.features_to_keep = keep

    def _transform(self, dataframe):
        
        features = self.get_features()
        out_col = self.getOutputCol()
        in_col = self.getInputCol()

        def feature_collect(data):
            lines = data.splitlines(keepends=False)

            filtered_lines = []
            for line in lines:
                json_line = json.loads(line)
                feature_array = json_line.get('features')
                filtered_feature_array = [feature 
                                          for feature 
                                          in feature_array 
                                          if feature.get('name') in self.features_to_keep]
                json_line['features'] = filtered_feature_array
                filtered_lines.append(json.dumps(json_line))

            return '\n'.join(filtered_lines)

        get_cntn = udf(feature_collect, StringType())
        return dataframe.withColumn(out_col, get_cntn(in_col))


