This directory will be populated when you run run_job.py (possibly by running slurm_jobs.sh). Each new model application corresponding to an element of the dictionary defined by model_params.py will get its own folder within this directory; that folder will have a name that matches the dictionary element. For example, after running the first and fourth jobs indicated by model_params.py, you might see:

mille
  out
    pretrain_1a
      checkpoint
      checkpoint_297.data-00000-of-00001
      checkpoint_297.index
      checkpoint_297.meta
      checkpoint_298.data-00000-of-00001
      checkpoint_298.index
      checkpoint_298.meta
      checkpoint_299.data-00000-of-00001
      checkpoint_299.index
      checkpoint_299.meta
    train_1a
      checkpoint
      checkpoint_255.data-00000-of-00001
      checkpoint_255.index
      checkpoint_255.meta
      checkpoint_256.data-00000-of-00001
      checkpoint_256.index
      checkpoint_256.meta
      checkpoint_257.data-00000-of-00001
      checkpoint_257.index
      checkpoint_257.meta
