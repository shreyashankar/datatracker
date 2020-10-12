# datatracker

WIP experimental project to make for a better ML development UX.

## Themes

* Easy reproducibility: template for custom makefile, and require all seeds to be defined in the makefile
* Easy to debug: ID every data point and track whatever the user wants on each iteration of training
* Easy to interact with: write signal handlers to inspect "states" and "transitions" of the modeling process
* Model-agnostic: a model is just a function, no matter what framework it was written in

## Tasks

- [x] Add a template makefile to make any ML project.
- [x] Add a `main.py` for the application-specific Python code.
- [x] Create `datatracker` abstraction to accept any tracking function as well as inputs.
- [ ] Figure out SIGSTP and SIGCONT abstractions.
- [ ] Log all params.
- [ ] Pipe `git diff` to some log.
- [ ] Figure out how to work with `pandas`, `tf.datasets`, and `torch.utils.data`.
- [ ] idk more stuff
