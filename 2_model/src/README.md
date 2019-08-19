## Code structure

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


## Testing the model

Below are some entry points to running example models.

For debugging the python code using any old data:
```py
import sys
sys.path.append('2_model/src')
from apply_pgnn import apply_pgnn
apply_pgnn(
        phase = 'pretrain',
        learning_rate = 0.008,
        state_size = 14, # Step 1: try number of input drivers, and half and twice that number
        ec_threshold = 24, # TODO: calculate for each lake: take GLM temperatures, calculate error between lake energy and the fluxes, take the maximum as the threshold. Can tune from there, but it usually doesn’t change much from the maximum
        dd_lambda = 0, # TODO: implement depth-density constraint in model
        ec_lambda = 0.025, # original mille lacs values were 0.0005, 0.00025
        l1_lambda = 0.05,
        data_file = '1_format/tmp/pgdl_inputs/nhd_10596466.npz',
        sequence_offset = 100,
        max_batch_obs = 50000,
        n_epochs = 2,
        min_epochs_test = 0,
        min_epochs_save = 2, # later is recommended (runs quickest if ==n_epochs)
        track_epoch_data = False, # False is faster, True is more interesting
        restore_path = '', #'2_model/out/EC_mille/pretrain',
        save_path = '2_model/tmp/test/nhd_10596466/pretrain')
```

For debugging the config-to-model pipeline:
```r
# start somewhere around model_config.R to create a config.tsv
# then run it through run_job.R
```
