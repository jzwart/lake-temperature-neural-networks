This directory has a file called model_params.py. That file should be run with python to create model_params.npy.

The mille/data directory should be populated locally with the Mille Lacs data, which are available at https://drive.google.com/drive/u/0/folders/1ZS8J68fakGfmJQiPJLY9D_yVUegGTPfc.

Then you can run a model.
* One entrypoint is apply_pgnn.py, which defines a function that will build and train a PGRNN on the specified dataset. This script in turn uses prep_data.py, tf_graph.py, tf_train.py, and (through tf_graph.py) physics.py to get its work done.
* A higher-level entrypoint is run_job.py, which will accept a command-line argument for the job ID to run, will look up that job's specifications in model_params.npy, and will run apply_pgnn() for those specifications.
* The highest-level entrypoint is slurm_jobs.sh, which defines a Slurm batch for running multiple run_job.py jobs.

The output will appear in mille/out, with one subfolder per model application (job).
