Image Captioning for Vietnamese



---

# **Image Captioning in Vietnamese using external knowledge for the Historical domain**

## 📌 **Project Description**
This repository provides the code and models for image captioning in Vietnamese using a transformer-based architecture. It leverages a pre-trained mBART model for generating captions from images.
---
##  **Dataset**
The dataset information will be updated soon. Please stay tuned for the data details and how to prepare it.

---
##  **Installation**
To set up the environment, you need to create and activate a conda environment using the provided icvn.yml file:
```sh
conda env create -n icvn --file vacnic.yml
conda activate icvn
```

###  **Train the Model**
For training, we recommend using an NVIDIA A100 GPU for optimal performance. The training process uses the mBART architecture, which is particularly well-suited for multilingual tasks.
```sh
run_full_train.sh
```
For more detail:
```sh
DATADIR --> the path to where you store the datasets
OUTPUTDIR --> the path to where you store the output models
```
Make sure to properly set these variables in the training script before running.
---

### **Evaluation**

By the end of our training code, we automatically generate a JSON file containing generated captions and ground truth captions.
For evaluating entities, please run:

```sh
python evaluate_entity.py
```
where:
```sh
DATADIR is the root dir for the datasets
OUTDIR is the output dir of your json file
```

Note that the package version needs to be changed. Please refer to the repo of **[Transform-and-Tell](https://github.com/alasdairtran/transform-and-tell)**
