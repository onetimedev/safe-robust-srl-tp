# Reproducing Our Results.
### Code and results for our paper: 
#### "Hybrid Safe Reinforcement Learning: Tackling Distribution Shift and Outliers with the Student-t's Process"
#### Steps to Run:
#### 1) Clone the repo onto your local machine
#### 2) Run the "setup_env.sh" shell script using "./setup_env.sh". Remember to follow installation prompts.
#### 3) Enter the main directory i.e., where the "Main.py" file is located.
#### 4) Reproducing the experiments detailed in the paper OR running the experiment driver with custom parameters
#### &nbsp;&nbsp;&nbsp; a) To reproduce the experiments run the shell script "experiment_driver.sh" using "./experiment_driver.sh"
#### &nbsp;&nbsp;&nbsp; b) To run an experiment with custom parameters run 
#### &nbsp;&nbsp;&nbsp; "python Main.py [experiment length] [outlier magnitude] [outlier ratio] [number of switch states] [record experiment data] [outliers in cost model] [outliers in reward model]" 
#### &nbsp;&nbsp;&nbsp; e.g. "python Main.py 250 40 0.2 2 True True True"
### Notes:
#### 1) Data discussed in the paper's data analysis section can be found in the "Data" directory
#### 2) If the "record experiment data" flag is set to True when running "Main.py" the data recorded can be found in the "ExperimentRuns" directory, with the experiment parameters used in the .csv filename.
#### 3) Tensorflow probability is designed to run on GPU(s) so please follow their instructions to installing the appropriate packages/drivers to do so. If your machine does not have a GPU, then the experiments can still run on the CPU, it will just take slightly longer. Please be patient! Tensorflow Probability GPU Installation Instructions: https://www.tensorflow.org/install/pip
