# Running models on Yeti

## Connecting to Yeti

It's pretty great to have ssh keys set up to communicate with Yeti. You have to do this for rsync anyway (see main README) but can take advantage of it to ssh into Yeti using
```sh
ssh yeti.cr.usgs.gov
```
from your terminal (Mac) or git bash shell (Windows).

If you don't want to set up SSH, you can still use PuTTy (Windows) to connect to yeti.cr.usgs.gov, logging in with your AD credentials.

## Environment configuration

I think we should install R packages in a local directory so we can all share the configuration. I propose /cxfs/projects/usgs/water/iidd/data-sci/lake-temp/Rlib. To point `.libPaths()` to that location, I've created an .Renviron file in the repo home directory containing
```sh
R_LIBS=/cxfs/projects/usgs/water/iidd/data-sci/lake-temp/Rlib
```

We'll need these packages installed:

- drake
- future
- future.batchtools

The igraph package is required for drake but won't compile using Yeti's default compiler, gcc v4.4.7. To install, I had to call `module load gcc/7.1.0` before running `R` and then `install.packages('igraph')` and `install.packages('drake')`. In subsequent yeti sessions, it seems it will be necessary to stick with `module load gcc/7.1.0` before trying to load the drake backage; otherwise, I get
```r
> library(drake)
Error: package or namespace load failed for ‘drake’ in dyn.load(file, DLLpath = DLLpath, ...):
 unable to load shared object '/cxfs/projects/usgs/water/iidd/data-sci/lake-temp/drake-test/Rlib/igraph/libs/igraph.so':
  /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by /cxfs/projects/usgs/water/iidd/data-sci/lake-temp/drake-test/Rlib/igraph/libs/igraph.so)
```
I also needed a more modern gcc to install data.table, which is a dependency of future.batchtools, so I stuck with 7.1.0 for that installation.

NB: I probably could have chosen one of the other available gcc modules, so long as the gcc version was at least 4.7 (https://stackoverflow.com/questions/54363300/r-compilation-error-iso-c-forbids-in-class-initialization-of-non-const-stati).
