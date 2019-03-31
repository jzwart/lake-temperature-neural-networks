This phase uses remake, drake, future, and future.batchtools to run models, as follows:
* 2_model.yml (in top project directory) - this governing remake file runs create_model_config() and run_model_tasks() once each
* config.R (in 2_model/src) - create a data_frame of configurations for all desired model runs
* model_tasks.R - build and execute a drake plan based on the configuration data_frame; write the .ind file expected by 2_model.yml if and when all tasks are up to date. uses file_in to depend on the data, restore, and python files, which means that whitespace changes to the python files are not ignored.
* slurm_batchtools.tmpl - template that drake uses each time it launches a model job. Resources are specified in the resources column of the drake plan in model_tasks.R
* run_job.R - call the tensorflow model via run_job.py

The core tensorflow model is defined in these files:
* run_job.py - the command-line entrypoint, which collects arguments from the command line and passes them to apply_pgnn.py
* apply_pgnn.py - master function that loads the data, builds the graph (using tf_graph.py), and trains the model (using tf_train.py)
* tf_graph.py - function to define the tensorflow graph including costs, gradients, optimizer, etc.; references physics.py
* physics.py - functions to calculate energy balance and depth-density relationships
* tf_train.py - function to train the tensorflow graph and save model coefficients and predictions
