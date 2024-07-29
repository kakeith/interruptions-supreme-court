"""
This file filters to justices based on the the min_num_chunks_per_just
(minimum number of junks per justice) set in config.yaml
"""
import sys, os
import glob
import json
import pprint
import pandas as pd 
import numpy as np 

from utils import *

def go_join_filter(df, config):
    """
    Makes all the necessary joins 
    And then filters to justices with the minimum number of chunks 
    (set in `config.yaml` as `min_num_chunks_per_just`)
    """

    # Join with the justice genders
    justice_gender_map = load_justice_gender()
    df['justice_gender'] = df.apply(lambda x: justice_gender_map[x['justice_name']], axis=1) 

    # Y: Token-normalized interruption rate 
    df['adv_interruption_rate'] = df.apply(lambda x: (1000*x['num_adv_utts_interrupted']/x['num_toks_adv']), axis=1)
    df['justice_interruption_rate'] = df.apply(lambda x: (1000*x['num_justice_utts_interrupted']/x['num_toks_justice']), axis=1)

    # A: Ideological alignment of an advocate and justice
    unk_advocates = []
    ideology_matches = []
    for i, row in df.iterrows(): 
        if (row['advocate_ideology'] not in ['conservative', 'liberal']): 
            if row['advocate_name'] not in unk_advocates:
                unk_advocates.append(row['advocate_name'])
            ideology_matches.append(np.nan)
        elif row['advocate_ideology'] == row['justice_ideology']: 
            ideology_matches.append(1)
        elif row['advocate_ideology'] != row['justice_ideology']:
            ideology_matches.append(0)
    assert len(ideology_matches) == len(df)
    ideology_matches = np.array(ideology_matches)

    #make the intersction between ideology and gender 
    df['adv_ideology_gender'] = [str(x)+'-'+str(y) for x,y in zip(df['advocate_gender'].to_numpy(), df['advocate_ideology'].to_numpy())]
    df['ideology_matches'] = ideology_matches

    #make dataset with the ideology_matches in [1, 0]
    df_final = df.copy().dropna()

    #make the integer variables 
    df_final['ideology_matches'] = [int(x) for x in df_final['ideology_matches']]

    print('original dataset num =', len(df))
    print('dataset w/ {0, 1} ideology mathces num =', len(df_final))

    #checks if justices have enough chunks and cases 
    df_by_just = df_final.groupby('justice_name').agg({'case_id': pd.Series.nunique, 
                                                    'utt_id_first': pd.Series.nunique, 
                                                    'advocate_name': pd.Series.nunique,
                                                    'advocate_gender': lambda x: x.value_counts().sort_index(),
                                                    'advocate_ideology': lambda x: x.value_counts().sort_index(),
                                                    'adv_ideology_gender': lambda x: x.value_counts().sort_index()})
    df_by_just.sort_values(by=['utt_id_first'], ascending=False)

    #only use justices that have >1000 unique chunks

    valid_justices = []
    for i, row in df_by_just.iterrows():
        if row['utt_id_first'] > config['min_num_chunks_per_just']: 
            valid_justices.append(row.name)

    print(f"Number of justices with >{config['min_num_chunks_per_just']} chunks", len(valid_justices))
    print("\t",valid_justices)

    # make the final datast with these justices 
    df = df_final.loc[df_final['justice_name'].isin(valid_justices)].copy()
    assert len(set(df['justice_name'].tolist())) == len(valid_justices)
    print('before justice filter, num chunks =', len(df_final))
    print('after justice filter, num chunks =', len(df))

    # KATIE TODO 
    if config['include_fem_issue'] == False: # exclude cases with "female issues"
        df = df[df["female_issue"] == 0]
        print("Exluded female issues, num chunks", len(df))

    #save the data frame 
    df.to_csv(config['final_df_path'], index=False)
    print("Saved final df to ->", config['final_df_path'])
    return df 


if __name__ == '__main__':
    config = load_config()
    pprint.pprint(config)
    df = load_chunks_df(config) # data frame after chucking
    go_join_filter(df, config) 