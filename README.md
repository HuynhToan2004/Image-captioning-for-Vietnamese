Image Captioning for Vietnamese
This repository provides the code and models for image captioning in Vietnamese using a transformer-based architecture. It leverages a pre-trained mBART model for generating captions from images.

Installation
To set up the environment, you need to create and activate a conda environment using the provided vacnic.yml file:

bash
Copy
Edit
conda env create -n icvn --file vacnic.yml
conda activate icvn
Data
The dataset information will be updated soon. Please stay tuned for the data details and how to prepare it.

Training
For training, we recommend using an NVIDIA A100 GPU for optimal performance. The training process uses the mBART architecture, which is particularly well-suited for multilingual tasks.

Key Directories
DATADIR: The path where you store the datasets.
OUTPUTDIR: The path where you store the output models.
Make sure to properly set these variables in the training script before running.

Evaluation
By the end of our training code, we automatically generate a JSON file containing generated captions and ground truth captions. The caption generation scores will be printed out and stored in the WandB log.

For evaluating entities, please run:

bash
Copy
Edit
python evaluate_entity.py
Where:

DATADIR is the root directory for the datasets.
OUTPUTDIR is the output directory for your JSON file.
Note that the package version needs to be changed. Please refer to the repo of **[Transform-and-Tell](https://github.com/alasdairtran/transform-and-tell)**

