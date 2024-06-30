import os
import io
import sys
from PIL import Image
import base64
from io import BytesIO
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from helmet.exception import HelmetException
from helmet.logger import logging
from helmet.configuration.s3_operations import S3Operation
from helmet.constants import *


class PredictionPipeline:
    def __init__(self):
        self.s3 = S3Operation()
        self.bucket_name = BUCKET_NAME

    def image_loader(self, image_bytes):
        """load image, returns cuda tensor"""
        logging.info("Entered the image_loader method of PredictionPipeline class")
        try:
            # image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image = Image.open(io.BytesIO(image_bytes))
            convert_tensor = transforms.ToTensor()
            tensor_image = convert_tensor(image)
            # image = image[:3]
            image_int = torch.tensor(tensor_image * 255, dtype=torch.uint8)
            logging.info("Exited the image_loader method of PredictionPipeline class")
            return tensor_image, image_int

        except Exception as e:
            raise HelmetException(e, sys) from e

    

    def get_model_from_s3(self) -> BytesIO:
        logging.info("Entered the get_model_from_s3 method of PredictionPipeline class")
        try:
            os.makedirs("artifacts/PredictModel", exist_ok=True)
            predict_model_path = os.path.join(os.getcwd(), "artifacts", "PredictModel", TRAINED_MODEL_NAME)
            
            # Reading the model data from S3
            self.s3.read_data_from_s3(self.bucket_name, TRAINED_MODEL_NAME, predict_model_path)
            
            # Check if the file exists and is not empty
            if not os.path.exists(predict_model_path) or os.path.getsize(predict_model_path) == 0:
                raise HelmetException(f"File {predict_model_path} is missing or empty.", sys)
            
            # Load the file into a buffer
            with open(predict_model_path, 'rb') as f:
                model_data = f.read()
            
            model_buffer = BytesIO(model_data)
            logging.info("Model loaded into buffer successfully.")
            return model_buffer

        except Exception as e:
            logging.error(f"Error in get_model_from_s3: {e}")
            raise HelmetException(e, sys) from e


    
    # def get_model_from_s3(self) -> str:
    #     """
    #     Method Name :   predict
    #     Description :   This method predicts the image.

    #     Output      :   Predictions
    #     """
    #     logging.info("Entered the get_model_from_s3 method of PredictionPipeline class")
    #     try:
    #         # Loading the best model from s3 bucket
    #         os.makedirs("artifacts/PredictModel", exist_ok=True)
    #         predict_model_path = os.path.join(os.getcwd(), "artifacts", "PredictModel", TRAINED_MODEL_NAME)
    #         best_model_path = self.s3.read_data_from_s3(TRAINED_MODEL_NAME, self.bucket_name, predict_model_path)
    #         logging.info("Exited the get_model_from_s3 method of PredictionPipeline class")
    #         return best_model_path

    #     except Exception as e:
    #         raise HelmetException(e, sys) from e

    def prediction(self, model_buffer: BytesIO, image_tensor, image_int_tensor) -> float:
        logging.info("Entered the prediction method of PredictionPipeline class")
        try:
            # Rewind the buffer before loading
            model_buffer.seek(0)
            
            # Load the model from buffer
            model = torch.load(model_buffer, map_location=torch.device(DEVICE))
            model.eval()
            with torch.no_grad():
                prediction = model([image_tensor.to(DEVICE)])
                pred = prediction[0]

            bbox_tensor = draw_bounding_boxes(image_int_tensor,
                                pred['boxes'][pred['scores'] > 0.8],
                                [PREDICTION_CLASSES[i] for i in pred['labels'][pred['scores'] > 0.8].tolist()],
                                width=4).permute(0, 2, 1)

            transform = transforms.ToPILImage()
            img = transform(bbox_tensor)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())

            logging.info("Prediction completed successfully.")
            return img_str

        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            raise HelmetException(e, sys) from e

    
    # def prediction(self, model_buffer: BytesIO, image_tensor, image_int_tensor) -> float:
    #     logging.info("Entered the prediction method of PredictionPipeline class")
    #     try:
    #         model = torch.load(model_buffer, map_location=torch.device(DEVICE))
    #         model.eval()
    #         with torch.no_grad():
    #             prediction = model([image_tensor.to(DEVICE)])
    #             pred = prediction[0]

    #         bbox_tensor = draw_bounding_boxes(image_int_tensor,
    #                             pred['boxes'][pred['scores'] > 0.8],
    #                             [PREDICTION_CLASSES[i] for i in pred['labels'][pred['scores'] > 0.8].tolist()],
    #                             width=4).permute(0, 2, 1)

    #         transform = transforms.ToPILImage()
    #         img = transform(bbox_tensor)
    #         buffered = BytesIO()
    #         img.save(buffered, format="JPEG")
    #         img_str = base64.b64encode(buffered.getvalue())

    #         logging.info("Exited the prediction method of PredictionPipeline class")
    #         return img_str

    #     except Exception as e:
    #         raise HelmetException(e, sys) from e


    
    
    def run_pipeline(self, data):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            image, image_int = self.image_loader(data)
            print(image.shape)
            print(image_int.shape)
            best_model_path: str = self.get_model_from_s3()
            detected_image = self.prediction(best_model_path, image, image_int)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return detected_image
        except Exception as e:
            raise HelmetException(e, sys) from e