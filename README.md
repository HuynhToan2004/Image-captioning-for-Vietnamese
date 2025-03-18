
\title{Image Captioning for Vietnamese}
\author{}
\date{}
\maketitle

This repository provides the code and models for image captioning in Vietnamese using a transformer-based architecture. It leverages a pre-trained mBART model for generating captions from images.

\section*{Installation}

To set up the environment, you need to create and activate a conda environment using the provided \texttt{vacnic.yml} file:

\begin{verbatim}
conda env create -n icvn --file vacnic.yml
conda activate icvn
\end{verbatim}

\section*{Data}

The dataset information will be updated soon. Please stay tuned for the data details and how to prepare it.

\section*{Training}

For training, we recommend using an NVIDIA A100 GPU for optimal performance. The training process uses the mBART architecture, which is particularly well-suited for multilingual tasks.

\subsection*{Key Directories}

\begin{itemize}
  \item \textbf{DATADIR}: The path where you store the datasets.
  \item \textbf{OUTPUTDIR}: The path where you store the output models.
\end{itemize}

Make sure to properly set these variables in the training script before running.

\section*{Evaluation}

By the end of our training code, we automatically generate a JSON file containing generated captions and ground truth captions. The caption generation scores will be printed out and stored in the WandB log.

For evaluating entities, please run:

\begin{verbatim}
python evaluate_entity.py
\end{verbatim}

Where:
\begin{itemize}
  \item \textbf{DATADIR} is the root directory for the datasets.
  \item \textbf{OUTPUTDIR} is the output directory for your JSON file.
\end{itemize}

Note that the package version needs to be changed. Please refer to the repo of \href{https://github.com/alasdairtran/transform-and-tell}{Transform-and-Tell} (also where you get the raw datasets from), and use the versions indicated in their repo.

