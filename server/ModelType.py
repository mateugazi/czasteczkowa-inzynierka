from redis_om import (Field, JsonModel)
from typing import List
from Parameter import Parameter

class ModelType(JsonModel):
  name: str = Field(index = True)
  identifier: str = Field(index = True)
  regression: bool = Field(index = False)
  parameters: List[Parameter]