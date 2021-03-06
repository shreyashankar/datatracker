# datatracker

WIP experimental project to make for a better ML development UX.

## Themes

* Easy reproducibility: template for custom makefile, and require all seeds to be defined in the makefile
* Easy to debug: ID every data point and track or do analysis on whatever the user wants on each iteration of training
* Easy to interact with: write signal handlers to inspect "states" and "transitions" of the modeling process
* Model-agnostic: a model is just a function, no matter what framework it was written in

## Tasks

- [x] Add a template makefile to make any ML project.
- [x] Add a `main.py` for the application-specific Python code.
- [x] Create `datatracker` abstraction to accept any tracking function as well as inputs.
- [x] Log biggest/smallest gains in any metric.
- [x] Log biggest/smallest gains in any metric at the iter level, not just globally.
- [ ] Track when an example goes from mispredicted to correctly predicted.
- [ ] Track when an example goes from correctly predicted to mispredicted.
- [x] Track most confident false positives and false negatives.
- [ ] Add support for val set (it already implicitly exists but we need to make sure they have different ids).
- [ ] Partially log metrics depending on the size of the dataset (not every iter, for example).
- [ ] Figure out SIGSTP and SIGCONT abstractions.
- [ ] Log all hparams.
- [ ] Pipe `git diff` to some log.
- [ ] Figure out how to work with `pandas`, `tf.datasets`, and `torch.utils.data`.
- [ ] idk more stuff
