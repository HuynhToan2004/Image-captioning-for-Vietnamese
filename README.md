Image Captioning for Vietnamese
🚀 Multilingual Captioning using mBART for Vietnamese Image Captioning

📌 Project Description
This project leverages deep learning to generate captions for images in Vietnamese using the mBART architecture, a transformer-based model fine-tuned for multilingual tasks. The model generates accurate and meaningful captions for Vietnamese images by training on a large dataset.

🗂 Dataset
The dataset information will be updated soon. Please stay tuned for the data details and how to prepare it.

📊 Model Performance
The model uses the mBART architecture for multilingual caption generation. We recommend using an NVIDIA A100 GPU for optimal performance.

🎯 Training Requirements
DATADIR: The path where you store the datasets.
OUTPUTDIR: The path where you store the output models.
⚙️ Installation & Usage
1️⃣ System Requirements
Python 3.8+
NVIDIA GPU (recommended)
Conda
2️⃣ Install Dependencies
Run the following command to set up the conda environment:

sh
Copy
Edit
conda env create -n icvn --file vacnic.yml
conda activate icvn
3️⃣ Train the Model
After setting up the environment, you can start training the model using the mBART architecture:

sh
Copy
Edit
python train.py --datadir /path/to/dataset --outputdir /path/to/output
📌 Technologies Used
Deep Learning Frameworks: PyTorch, Transformers
Pre-trained Models:
mBART (for multilingual caption generation)
Training Optimization: Adam Optimizer, PyTorch DataLoader
Evaluation Metrics: BLEU, METEOR, ROUGE
🚀 Future Improvements
🛠 Improve model accuracy by fine-tuning hyperparameters and adding more training data.
🛠 Expand the dataset for better generalization.
🤝 Contributions
We welcome contributions! To contribute:

Fork this repository
Create a new branch: git checkout -b feature-new
Commit your changes: git commit -m "Add new feature"
Push to GitHub: git push origin feature-new
Create a Pull Request
📌 Evaluation
By the end of our training code, we automatically generate a JSON file containing generated captions and ground truth captions. The caption generation scores will be printed out and stored in the WandB log.

For evaluating entities, please run:

sh
Copy
Edit
python evaluate_entity.py
Where:

DATADIR is the root directory for the datasets.
OUTPUTDIR is the output directory for your JSON file.
Note that the package version needs to be changed. Please refer to the repo of Transform-and-Tell (also where you get the raw datasets from), and use the versions indicated in their repo.


