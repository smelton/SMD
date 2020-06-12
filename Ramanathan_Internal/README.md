This version of the code is specifically for the Harvard Odyssey cluster. It will run 20 jobs, and compile the results at the end.

A few changes need to be made to run:

1. Make sure your data is in the same directory, and adapt 'array_counts.py' such that your data is loaded properly on line 15-16

2. Adjust the parameters in array_counts.py as desired.

3.Change lines 10-12 of BOTH bash scripts to activate the appropriate conda environment (which must contain numpy and sklearn).

4. Collect all of these files in the same directory

5. Run in the command line (in the same directory)

sbatch --array=0-19 counts.sh
Submitted batch job <jobID>


This starts an array of jobs which construct cluster proposals and save the outputs in the same directory. It will return a jobID, which you copy and input into the next command

sbatch --dependency=afterok:<jobID> collect.sh

The result will be a Z_score.npy file that has Z_scores for each dimension
