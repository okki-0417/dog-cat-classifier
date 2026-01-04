from dataclasses import dataclass


@dataclass
class BreedPrediction:
    name: str
    probability: float


@dataclass
class ClassificationResult:
    dog_probability: float
    cat_probability: float
    top_dog_breed: BreedPrediction
    top_cat_breed: BreedPrediction

    def is_dog(self) -> bool:
        return self.dog_probability > self.cat_probability

    def is_cat(self) -> bool:
        return self.cat_probability > self.dog_probability
