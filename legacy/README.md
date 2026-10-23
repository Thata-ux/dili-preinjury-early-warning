# Superseded analysis

`analysis_pipeline.ipynb` in this folder is the pipeline from the earlier version of
this work, in which hepatotoxicity was classified from expression measured at the *same*
timepoint as the histopathology that defined the label.

It is kept for transparency, not for use. That design measures concurrent detection of
established injury rather than prediction of injury yet to occur, and on this resource
administered dose alone reaches an AUC of 0.893 on that task — so a high score there
carries little information about early-warning value. Re-evaluating it under
compound-held-out validation drops performance from 0.952 to 0.848, below the raw-dose
baseline of 0.893.

The current analysis in the repository root replaces it: predictors restricted to
histopathologically normal tissue, outcomes taken from later timepoints, whole compounds
held out, and dose specified in advance as the baseline.
