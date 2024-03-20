# Running the SARSP Algorithm
### Code and results for our paper: 
#### "Robust Hybrid Safe Reinforcement Learning: Distribution Shift and Outliers Using the Student-t Process"
#### Steps to Run:
#### 1) Clone the repo onto your local machine
#### 2) Run the "setup_env.sh" shell script using "./setup_env.sh". Remember to follow installation prompts.
#### 3) Enter the main directory i.e., where the "Main.py" file is located.
#### 4) Reproducing the experiments detailed in the paper OR running the experiment driver with custom parameters
#### &nbsp;&nbsp;&nbsp; a) To reproduce the experiments run the shell script "experiment_driver.sh" using "./experiment_driver.sh"
#### &nbsp;&nbsp;&nbsp; b) To run an experiment with custom parameters run 
#### &nbsp;&nbsp;&nbsp; "python Main.py [experiment length] [outlier magnitude] [outlier ratio] [number of switch states] [record experiment data] [outliers in cost model] [outliers in reward model]" 
#### &nbsp;&nbsp;&nbsp; e.g. "python Main.py 250 40 0.2 2 True True True"
