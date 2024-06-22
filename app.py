
from helmet.logger import logging
from helmet.exception import HelmetException
import sys
from helmet.pipeline.training_pipeline import TrainPipeline

 
train_pipeline = TrainPipeline()
train_pipeline.run_pipeline()
print("success")