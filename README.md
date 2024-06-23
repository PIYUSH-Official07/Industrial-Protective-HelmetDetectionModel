# Industrial-Protective-HelmetDetectionModel

#### Language and Libraries

<p>
<a><img src="https://img.shields.io/badge/Python-2C2D72?style=for-the-badge&logo=python&logoColor=white" alt="python"/></a>
<a><img src="https://img.shields.io/badge/Pandas-FFA07A?style=for-the-badge&logo=pandas&logoColor=darkgreen" alt="pandas"/></a>
<a><img src="https://img.shields.io/badge/Numpy-8774b8?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy"/></a>
<a><img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="opencv"/></a>
<a><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="pytorch"/></a>
<a><img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)" alt="docker"/></a>
<a><img src="https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)" alt="aws"/></a>
</p>


## Problem statement
This project focuses on detecting the presence of industrial protective helmets worn by workers in various industrial environments using computer vision techniques implemented with PyTorch. Ensuring that workers wear helmets is critical for maintaining workplace safety and adhering to safety regulations. The primary objective of this project is to develop a pipeline that can accurately determine whether a person is wearing a helmet or not.

## Solution Proposed
To address the problem of helmet detection, we have implemented a solution leveraging advanced computer vision techniques. Utilizing the PyTorch framework, we developed a custom object detection network specifically designed for identifying helmets. Subsequently, An API was developed to process images and predict whether a person is wearing a helmet or not. The entire application was then containerized using Docker and deployed on the AWS cloud.

## Dataset Used

This dataset, contains 5000 images with bounding box annotations in the PASCAL VOC format for these 3 classes:
Helmet;
Person;
Head; 
The primary objective is to ascertain whether individuals are wearing helmets.
Dataset link: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
## How to run?

### Step 1: Clone the repository
```bash
git clone my repository 
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -p env python=3.8 -y
```

```bash
conda activate env
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Export the  environment variable
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

```
Before running server application make sure your `s3` bucket is available and empty

### Step 5 - Run the application server
```bash
python app.py
```

### Step 6. Train application
```bash
http://localhost:8080/train
```

### Step 7. Prediction application
```bash
http://localhost:8080
```

## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image

```
docker build --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION> . 

```

3. Run the Docker image

```
docker run -d -p 8080:8080 <IMAGEID>
```

👨‍💻 Tech Stack Used
1. Python
2. Pytorch
3. FastAPI
4. Docker
5. Computer vision

🌐 Infrastructure Required.
1. AWS S3
2. AWS ECR
3. AWS EC2
4. Git Actions


## `helmet` repo is the source package folder which contains 

**Artifact** : Stores all artifacts created from running the application


**Components**: This section includes all the integral elements of the Machine Learning Project, each playing a crucial role in the workflow:

Data Ingestion: Responsible for acquiring and importing data from various sources into the project.
Data Transformation: Handles the preprocessing and transformation of raw data into a suitable format for model training.
Model Trainer: Engages in training machine learning models using the prepared datasets.
Model Evaluation: Conducts comprehensive evaluation of the trained models to assess their performance and accuracy.
Model Pusher: Manages the deployment of the trained and validated models into a production environment.

**Custom logger and exception** handling mechanisms have been integrated into the project to enhance debugging and streamline error management.


## Conclusion




=====================================================================