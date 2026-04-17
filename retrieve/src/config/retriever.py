from typing import Optional

import pydantic
import yaml

from .base import EnvYaml

class DatasetYaml(pydantic.BaseModel):
    name: str
    text_encoder_name: str
    use_llm_extracted_entities: bool = False  # 新增，默认值 False
    use_kg_enhanced: bool = False             # 新增，默认值 False
    entity_mapping_file: Optional[str] = None    # 新增，可选字段
    data_base_dir: Optional[str] = None          # 新增，可选字段

class DDEYaml(pydantic.BaseModel):
    num_rounds: int
    num_reverse_rounds: int

class RetrieverYaml(pydantic.BaseModel):
    topic_pe: bool
    DDE_kwargs: DDEYaml

class OptimizerYaml(pydantic.BaseModel):
    lr: float

class EvalYaml(pydantic.BaseModel):
    k_list: str

class RetrieverExpYaml(pydantic.BaseModel):
    num_epochs: int
    patience: int
    save_prefix: str

class RetrieverTrainYaml(pydantic.BaseModel):
    env: EnvYaml
    dataset: DatasetYaml
    retriever: RetrieverYaml
    optimizer: OptimizerYaml
    eval: EvalYaml
    train: RetrieverExpYaml

def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    task = yaml_data.pop('task')
    assert task == 'retriever'
    
    config = RetrieverTrainYaml(**yaml_data).model_dump()
    config['eval']['k_list'] = [
        int(k) for k in config['eval']['k_list'].split(',')]

    return config
