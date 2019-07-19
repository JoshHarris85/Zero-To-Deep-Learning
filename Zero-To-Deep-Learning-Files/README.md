# Welcome to the Zero to Deep Learning Book repository

<p align="center">
  <img src="assets/cover.png" alt="Zero to Deep Learning Book" width="306" height="396"/>
</p>

Hello and welcome to the Zero to Deep Learning code repository.

This code is associated with the book [Zero to Deep Learning](www.zerotodeeplearning.com), written by Francesco Mosconi and published by Fullstack.io.


With this course you will:
* Understand the fundamentals of Deep Learning
* Build Fully Connected Neural Networks to solve Classification and Regression
* Train Convolutional Neural Networks to recognize images
* Design Recurrent Neural Networks to forecast time series and classify text
* Discover Embeddings to represent categorical variables with high cardinality
* Leverage Dropout & Batch Norm to improve your model performance
* Use GPUs in the cloud to train your models faster
* Piggyback on previous work with transfer learning
* Deploy your models to a cloud server using Tensorflow

You will also solve several practical problems and applications:
* Predict the price of a house
* Diagnose diabetes and detect fake banknotes
* Classifying images of different sports
* Classify the sentiment in a sentence
* Forecast future energy consumption and the price of Bitcoin
* Deploy an API that predicts phone location from wifi signal intensity
* and much more ...


## Repository Content

* [README.md](./README.md): This file
* [environment.yml](./environment.yml): The file that specifies the packages required to run the notebooks.
* [environment-gpu.yml](./environment-gpu.yml): Same as environment.yml but with GPU support.
* [course](./course): The main folder of the course. It contains all the Jupyter notebooks used in the course.
* [solutions](./solutions): The solutions folder. It contains the Jupyter notebooks with solutions to each exercise.
* [data](./data): Folder containing all the datasets used in the notebooks. When a dataset is too big, a script is provided to download it instead of the actual data.


## Getting started guide
The code is provided as a set of Jupyter Notebooks. If you are familiar with Jupyter, simply check that you have installed the correct dependencies provided in the [environment.yml](./environment.yml) file. If you are just getting started, follow the next steps to get the environment.


#### Install Python 3.7
We recommend using the [Anaconda Python distribution](www.anaconda.com). You can install the full distribution or you can install just the minimum required packages using [Miniconda](https://conda.io/en/latest/miniconda.html). In either case, make sure to install Python 3.7.

#### Go to the course folder
Open a terminal and change directory to the repository you have just downloaded. This step depends on the system you are using. On Mac and Linux you can open a bash terminal, on Windows you will open the Anaconda Prompt. Once you've opened the prompt, just type:

```
cd zerotodeeplearning
```

#### Create and activate the course environment
Our code relies on a set of dependencies listed in the [environment.yml](./environment.yml) file provided. We create an environment that contains these dependencies with the command:

```
conda env create
```

This will take a little bit of time. Please wait for the environment to create and once it's ready activate the environment. The command to activate the environment will be different on Mac/Linux and on Windows.

##### Activate the environment (Mac/Linux)
```
conda activate ztdlbook
```

##### Activate the environment (Windows)
```
activate ztdlbook
```

In either case, check that your prompt changed to:

```
(ztdlbook) $
```

#### GPU environment
We provide an alternate [environment-gpu.yml](./environment-gpu.yml) file, ready for GPU support. To create an environment with this file simply type:

```
conda env create -f environment-gpu.yml
```
so that conda doesn't default to the standard environment file.

#### Launch Jupyter Notebook
Once you have created and activated the environment, it's time to launch Jupyter notebook. In the same prompt type:

```
jupyter notebook
```

This will launch the Jupyter notebook server and open your default browser to show the current folder.

#### Open the first notebook
Go to the course folder, open the notebook `1_Getting_Started.ipynb` and run it.

You are good to go! Enjoy!


#### Troubleshooting installation
If for some reason you encounter errors while running the first notebook, the simplest solution is to delete the environment and start from scratch again.

To remove the environment:

- close the browser and go back to your terminal
- stop jupyter notebook `(CTRL-C)`
- deactivate the environment (Mac/Linux):

```
conda deactivate ztdlbook
```

- deactivate the environment (Windows 10):

```
deactivate ztdlbook
```

- delete the environment:

```
conda remove -y -n ztdlbook --all
```

- restart from environment creation and make sure that each steps completes till the end.

#### Updating Conda

One thing you can also try is to update your conda executable. This may help if you already had Anaconda installed on your system.

```
conda update conda
```

These instructions have been tested on:

- Mac OSX Sierra 10.12.4
- Ubuntu 16.04
- Windows 10


## Contributing and getting Help
If you're having trouble getting this project running, feel free to [open an issue](https://github.com/zerotodeeplearning/ztdl-keras-tensorflow-book/issues) on the open-source repository or [email us](mailto:help@zerotodeeplearning.com)!


## Copyright
Copyright 2017 Catalit LLC, all rights reserved.
Distribution of full notebooks and solutions is prohibited.
For an open-source version of the code only, check: https://github.com/zerotodeeplearning/ztdl-keras-tensorflow-book/.
