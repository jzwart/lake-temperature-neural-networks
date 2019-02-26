# lake-temperature-neural-networks
Pipeline #3 

This project Uses rsync and ssh to pull data/predictions from Yeti in the 1_format phase. For this to work, SSH keys must be set up for communication with Yeti:

* If you don't yet have a local ssh key pair, use `ssh-keygen -t rsa` from within a local terminal.

* Copy the public key to Yeti with `ssh-copy-id username@yeti.cr.usgs.gov` (also from within your local terminal). You can then check that you're set up by running `ssh username@yeti.cr.usgs.gov` from a terminal - it should log
you in without a password.

* On Windows with RStudio, there will be a problem in that SSH/rsync assume your .ssh folder is at `~/.ssh`, but `~` means `C:/Users/username` within a terminal but `C:/Users/username/Documents` within RStudio. Therefore you should create a symlink for the `.ssh` folder by calling `ln -s ~/.ssh ~/Documents/.ssh` in a bash shell.
