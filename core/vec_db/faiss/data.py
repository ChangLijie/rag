from typing import List

from datasets import Dataset, load_dataset
from pydantic import BaseModel, model_validator


class DataFormat(BaseModel):
    images: List[str]
    describe: List[str]

    @model_validator(mode="after")
    def check_equal_length(self) -> None:
        if len(self.images) != len(self.describe):
            raise ValueError("Length of images and describe lists must be the same")


class Process:
    """Deal data"""

    def __init__(self, raw_data: dict) -> None:
        self.dataset = self._turn2datasets(data=DataFormat(**raw_data))

    def _turn2datasets(self, data: dict) -> Dataset:
        """
        Turn type to Dataset

        Args:
            data (dict): Raw data

        Returns:
            Dataset: Dataset type
        """

        return Dataset.from_dict(data.dict())

    def save(self, file_path: str) -> None:
        """
        Save data.

        Args:

            file_path (str): path to save data
        """
        self.dataset.save_to_disk(file_path)
