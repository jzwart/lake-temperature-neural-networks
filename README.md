# lake-temperature-neural-networks
Pipeline #3 

It is assumed in the current code that processing will happen in different places for different analysis phases.

* Production of data for drivers, geometry, glm_preds, and obs: Pipelines 1 and 2, anywhere

* Pulling drivers, geometry, glm_preds, and obs into 1_format/tmp: 1_format phase, local computer

* Munging drivers, geometry, glm_preds, and obs into training-ready python data files in 1_format/tmp/pgdl_inputs: 1_format phase, local

* Pushing training-ready pgdl_inputs to yeti: 1_format phase, local

* Using pgdl_inputs to train and test models: 2_model phase, Yeti cluster

This project uses rsync and ssh to pull data/predictions from Yeti in the 1_format phase. For this to work, SSH keys must be set up for communication with Yeti:

* If you don't yet have a local ssh key pair, use `ssh-keygen -t rsa` from within a local terminal.

* Copy the public key to Yeti with `ssh-copy-id username@yeti.cr.usgs.gov` (also from within your local terminal). You can then check that you're set up by running `ssh username@yeti.cr.usgs.gov` from a terminal - it should log
you in without a password.

* On Windows with RStudio, there will be a problem in that SSH/rsync assume your .ssh folder is at `~/.ssh`, but `~` means `C:/Users/username` within a terminal but `C:/Users/username/Documents` within RStudio. Therefore you should create a symlink for the `.ssh` folder by calling `ln -s ~/.ssh ~/Documents/.ssh` in a bash shell.
