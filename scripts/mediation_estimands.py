"""
This file contains code for estimands in causal mediation 
"""
import pandas as pd
import numpy as np

def make_df_justice(justice_name, df, mediator_colm_name):
    """
    For a single justice, creates the "cannonical" dataframe with columns M, T, Y 

    Inputs: 
        -justice_name (str): justice name 
        - df (pd.DataFrame): finalized data frame 
        - mediator_colm_name (str): mediator of interest  
    """ 
    df1 = df.loc[df['justice_name']==justice_name]
    
    #rename the columns for Pearl's cannonical equations 
    mapping = {mediator_colm_name: 'M', 
                'advocate_gender': 'T',
                'interruption_rate':'Y'}
    df1 = df1.rename(columns=mapping)

    #make male=0, female = 1
    df1["T"]=df1["T"].map({'M': 0, 'F': 1})
    
    df2 = df1[[str(x) for x in mapping.values()]]
    return df2

def make_df_justice_general(justice_name, df, mediator_colm_name, treatment_colm_name, outcome_colm_name):
    """
    For a single justice, creates the "cannonical" dataframe with columns M, T, Y 

    Inputs: 
        -justice_name (str): justice name 
        - df (pd.DataFrame): finalized data frame 
        - mediator_colm_name (str): mediator of interest  
    """ 
    df1 = df.loc[df['justice_name']==justice_name]
    
    #rename the columns for Pearl's cannonical equations 
    mapping = {mediator_colm_name: 'M', 
                treatment_colm_name: 'T',
                outcome_colm_name:'Y'}
    df1 = df1.rename(columns=mapping)
    
    if(treatment_colm_name=="advocate_gender"):
        #make male=0, female = 1
        df1["T"]=df1["T"].map({'M': 0, 'F': 1})
        
    df2 = df1[[str(x) for x in mapping.values()]]
    return df2

def make_df_justice_no_mediators(justice_name, df, treatment_colm_name, outcome_colm_name):
    """
    For a single justice, creates the "cannonical" dataframe with columns T, Y 

    Inputs: 
        - justice_name (str): justice name 
        - df (pd.DataFrame): finalized data frame 
    """ 
    df1 = df.loc[df['justice_name']==justice_name]
    
    mapping = {treatment_colm_name: 'T',
                outcome_colm_name:'Y'}
    df1 = df1.rename(columns=mapping)
    
    if(treatment_colm_name=="advocate_gender"):
        #make male=0, female = 1
        df1["T"]=df1["T"].map({'M': 0, 'F': 1})
        
    df2 = df1[[str(x) for x in mapping.values()]]
    return df2


def total_effect(df_just, justice_name):
    """
    Returns the total (naive effect)
    E[Y|T=1] - E[Y|T=0]
    """
    # E[Y|T=0]
    t0_rows = df_just.loc[df_just["T"] == 0]
    y0 = np.mean(t0_rows["Y"].to_numpy())

    # E[Y|T=1]
    t1_rows = df_just.loc[df_just["T"] == 1]
    y1 = np.mean(t1_rows["Y"].to_numpy())

    # TE
    te = y1 - y0
    return {"justice_name": justice_name, "te": te, "y0": y0, "y1": y1, "n_t0": len(t0_rows), "n_t1": len(t1_rows)}


def pearls_mediation(df_just, justice_name):
    """
    Calculates the Natural Direct Effect (NDE) and Natural Indirect Effect (NIE)
    and returns intermediary values as well (e.g. P(M=m|T=0))

    Pearl, Causal Inference in Statastics

    Eqns. 4.51 and 4.52
    $$ NDE = \sum_m [E[Y|T=1, M=m] - E[Y|T=0, M=m]]\cdot P(M=m|T=0)$$
    $$ NIE = \sum_m [E[Y|T=0, M=m]] \cdot [P(M=m|T=1)- P(M=m|T=0)]$$

    where

    $$ P(M=m|T=0) = \frac{\text{count\_chunks}(M=m \cap T=0)}{\text{count\_chunks}(T=0)} $$

    Inputs:
        df_just: (pd.DataFrame) dataframe for that particular justice with columns M, T, Y
        justice_name: (str) justice's name for printout
    """
    out = {}
    out["justice_name"] = justice_name

    #TODO: put in warning that this is only for discrete M's 
    m_support = set(df_just["M"].values)

    # subset based on ms
    m2df = {}
    for m in m_support:
        mdf = df_just.loc[df_just["M"] == m]
        m2df[m] = mdf

    nde = 0
    nie = 0

    # sum for each m
    for m in m_support:
        # P(M=m|T=0)
        t0_rows = df_just.loc[df_just["T"] == 0]
        m_cond_t0_rows = t0_rows.loc[t0_rows["M"] == m]
        mt0 = len(m_cond_t0_rows) / len(t0_rows)
        out[f"P(M={m}|T=0)"] = mt0
        out[f"n_T=0,M={m}"] = len(m_cond_t0_rows)

        # P(M=m|T=1)
        t1_rows = df_just.loc[df_just["T"] == 1]
        m_cond_t1_rows = t1_rows.loc[t1_rows["M"] == m]
        mt1 = len(m_cond_t1_rows) / len(t1_rows)
        out[f"P(M={m}|T=1)"] = mt1
        out[f"n_T=1,M={m}"] = len(m_cond_t1_rows)

        # E[Y|T=0, M=m]
        yt0m = np.mean(m_cond_t0_rows["Y"])
        out[f"E[Y|T=0,M={m}]"] = yt0m

        # E[Y|T=1, M=m]
        yt1m = np.mean(m_cond_t1_rows["Y"])
        out[f"E[Y|T=1,M={m}]"] = yt1m

        # calculate the sums for NDE and NIE
        nde_m = (yt1m - yt0m) * mt0
        nde += nde_m
        out[f"NDE(M={m})"] = nde_m

        nie_m = yt0m * (mt1 - mt0)
        nie += nie_m
        out[f"NIE(M={m})"] = nie_m

    out["nde"] = nde
    out["nie"] = nie
    return out
