# "Let Me Just Interrupt You": Estimating Gender Effects in Supreme Court Oral Arguments

This is replication code to support the paper ["Let Me Just Interrupt You": Estimating Gender Effects in Supreme Court Oral Arguments]() which forthcoming in *Journal of Law and Courts.* 

If you use this code or data, please cite our paper: 

```
@article{cai2024interrupt,
  author    = {Cai, Erica and Gupta, Ankita and Keith, Katherine A., and O'Connor, Brendan and Rice, Douglas},
  title     = {“Let Me Just Interrupt You”: Estimating Gender Effects in Supreme Court Oral Arguments},
  journal   = {Journal of Law and Courts},
  year      = {2024},
  note      = {Forthcoming}
}
```

## Set-up 

Follow these instructions to set-up your repository. You will need to download [Anaconda](https://www.anaconda.com/) to run `conda` and `python` commands. 

```
git clone git@github.com:kakeith/interruptions-supreme-court.git
cd interruptions-supreme-court
conda create -y --name scourt python==3.9
conda activate scourt
pip install -r requirements.txt
```
## Raw Data 

The raw data in the `raw_data` folder come from the following sources: 

1. The [Supreme Court Data Base](http://scdb.wustl.edu/). 

2. [ConvoKit's Supreme Court Oral Arguments Corpus](https://convokit.cornell.edu/documentation/supreme.html) (which sources from [Oyez](https://www.oyez.org/)). 

3. [Rafo et al.'s World Gender Name Dictionary](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YPRQH8# )


4. To create `justice-ideology.txt`, we measure ideology using the average (arithmetic mean) of the justice’s time-varying [Martin-Quinn score](http://mqscores.wustl.edu/) (Martin and Quinn 2002), and treat any values less than zero as a liberal justice, and values greater than zero as a conservative justice.

5. The file `backchannel.txt` are phrasal backchannel cues that the authors curated. 

## Code pipelne 

Pipeline descisions are detailed in `scripts/config.yaml`. To replicate our code pipeline and analysis, run the following scripts in order 

1. For pre-processing and chunking, 

	```
	cd scripts/ 
	python create_analyze_chunks.py
	python filter.py 
	```

	This takes about 15-20 minutes to run on our machine. 

2. For the main analysis and plots in our paper, run all cells in the following jupyter notebook  

	```
	scrips/analysis.ipynb
	```
	
3. To make "Figure 5: Justice Interruption Rates (y-axis) by Martin & Quinn Ideology Scores (x-axis)", run `scripts/interruptionsPlot.r` using R. 

4. For supplementary and corroborative analyses run 

	```
	scripts/supplemental_analysis.ipynb
	```

5. To obtain the results in the Appendix for the backchannel cue removal, use the configuration specified in `scripts/config-backchannels.yaml` and re-run the pipeline (`create_analyze_chunks.py`, `filter.py` and `analysis.ipynb`). 


## Notes

In the ConvoKit/Ozez data there are still errors with `John G. Roberts Jr.` when he was an advocate. This results in warnings after running `create_analyze_chunks.py` such as  `John G. Roberts Jr.  not found in caseid2stuff dict, assigning unknown. case id: 1991_90-6531`. This warning should not substantively affect the results. 



