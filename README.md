In the project i made a notebook for a signature verification task where the model achieved a respectfully 80%-85% both in validation and testing

The modell used data augmentation and contrastive loss for the loss function with a 1.2 margin value.

Needed environment:

You have to download the anaconda or miniconda distribution, then execute these commands:

    1. Create a Conda environment:
        conda create --name tf_gpu python=3.10
    2. Activate the environment:
        conda activate tf_gpu
    3. Install CUDA and cuDNN:
        conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
    4. Install TensorFlow:
        pip install tensorflow==2.10

Needed libraries:

   pip install notebook keras opencv-python matplotlib scikit-image seaborn scikit-learn

If the project still doesn't use the GPU, follow these steps:

    1. Download CUDA 11.2 and cuDNN 8.1 from the official websites.
    2. After installing CUDA, copy the cuDNN files (bin, include, lib) into the appropriate CUDA directories:
        - bin -> C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin
        - include -> C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\include
        - lib -> C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\lib
    3. Add the following to your system's environment variables:
        - CUDA_HOME: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2
        - CUDA_PATH: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2
        - CUDA_PATH_V11_2: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2
    4. Restart the PC to apply changes.

