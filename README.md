# AMopt
Topology optimization for additive manufacturing

Specifically, this project aims to optimize a cube to minimize weight (and 
therefore cost) subject to the constraint that the compressive strength be
sufficient to support a given load.

# Requirements & Setup
TODO: Setup & Installation
* GMSH
* SfePy

To install python dependencies: 

```
python -m pip install -r requirements.txt
```

NOTE: This program was developed and tested only on Linux.

# Execution
Configure all the settings in settings.py, then run:

```
python topology.py
```

For running long optimizations (on linux), consider running it like this:

```
nohup python -u topology.py &> log.txt &
```

# Contributing to this repository with GIT
Please configure user.name and user.email (See Your Identity section of 
[this guide](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)) 
before making *any* commits.

For now, in order to simplify things and avoid any merge conflicts, just try
to avoid editing any files that you didn't create yourself. If you're going to
edit a file that you didn't create, that's probably a good time to [make a 
branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging).
