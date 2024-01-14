from redis_om import (Field, JsonModel)
from typing import List
from Parameter import Parameter

class ModelType(JsonModel):
  name: str = Field(index = True)
  parameters: List[Parameter]