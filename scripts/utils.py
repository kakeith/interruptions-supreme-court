"""
This script holds utility functions for the pipeline 
"""
import json
import datetime 
import numpy as np
import yaml 
import glob
import pandas as pd 
import math
import re
from tqdm import tqdm 

def parse_first_name(name):
    ss = name.split(" ")
    return ss[0].lower()

def parse_last_name(name):
    name = name.strip()
    suffix = name.split(' ')[-1] 
    if 'Jr.' in suffix:
        suffix =  name.split(' ')[-2]
    return suffix

def one_utt_rule_speech_disfluency(toks, disfluency_symbols=['--', '...'], single_dash=True):
    """
    Args: 
        - toks (list): list of words that have been tokenized
        - disfluency_symbols (list, optional): list of the disfluency symbols we're matching against
        - single_dash (bool, optional): If true, then we want to pick up disfluency examples from some transcripts which look like: 
            "Rush pu- Prudential HMO, Inc"
            which have a dash and a space 
        
    Output: 
        count of the number of times a text has 
            "WORD1 -- WORD1"
            
            e.g. "it -- it seems like -- like" would return 2 

    Note: prior to 2008 it looks like they use ... for disfluency symbols
        Example https://www.oyez.org/cases/2005/04-1506
    """
    #dont look at the final token, this is interruptions
    all_disfl = [tok for tok in toks[0:len(toks)-1] if tok in disfluency_symbols]
    num_disfluencies = len(all_disfl)

    #now go back thru and count the single dashes 
    if single_dash:
        has_dash = [tok for tok in toks[0:len(toks)-1] if tok[-1] == '-' and tok not in disfluency_symbols]
        num_disfluencies += len(has_dash) 

    return num_disfluencies

def load_justice2start_date(): 
    """
    manually coded via: https://en.wikipedia.org/wiki/List_of_justices_of_the_Supreme_Court_of_the_United_States
    """
    justice2start_date = {
    "Reed": "1938-01-25",
    "Douglas": "1939-04-17",
    "Frankfurter": "1939-01-30",
    "Black": "1937-08-19",
    "Clark": "1949-09-18", 
    "Minton": "1949-10-04", 
    "Warren": "1954-03-01",
    "Harlan": "1955-03-16",
    "Brennan": "1956-10-16", 
    "Whittaker": "1957-03-25", 
    "Stewart": "1958-10-14",
    "White": "1962-04-16", 
    "Goldberg": "1962-10-01",
    "Fortas": "1965-10-04", 
    "Marshall": "1967-10-02",
    "Burger": "1969-06-23", 
    "Blackmun": "1970-06-09", 
    "Powell": "1972-01-07", 
    "Rehnquist": "1972-01-07", 
    "Stevens": "1975-12-19", 
    "O'Connor": "1981-09-25", 
    "Scalia": "1986-11-26", 
    "Kennedy": "1988-02-18", 
    "Souter": "1990-10-09", 
    "Thomas": "1991-10-23", 
    "Ginsburg": "1993-09-10", 
    "Breyer": "1994-09-03", 
    "Roberts": "2005-09-29", 
    "Alito": "2006-01-31", 
    "Sotomayor": "2009-09-08", 
    "Kagan": "2010-09-07", 
    "Gorsuch": "2017-04-10", 
    "Kavanaugh": "2018-10-06", 
    "Barrett": "2020-10-27"
    }
    return justice2start_date

def load_chiefjustice2start_date(): 
    """
    manually coded via: https://en.wikipedia.org/wiki/List_of_justices_of_the_Supreme_Court_of_the_United_States
    """
    chiefjustice2start_date = {
    "Burger": "1969-06-09", 
    "Rehnquist": "1986-09-17", 
    "Roberts": "2005-09-29"
    }
    return chiefjustice2start_date

def is_chiefjustice_speaking(case_id, caseid2stuff, utt):
    """
    Sometimes justices who were previously advocates get
    labeled incorrectly depending on the year
    
    If the case_decide_date < justice_start_date
        the justice was an advocate 
    """ 
    speaker_type = utt.speaker.meta['type']

    #only need to look at false positives for justices 
    if speaker_type != "J":
        return False

    chiefjustice2start_date = load_chiefjustice2start_date()

    #make justice last name
    last_name = get_justice_last_name(utt)

    if (last_name not in ["Roberts","Rehnquist","Burger"]):
        return False 
    
    if chiefjustice2start_date.get(last_name) == None:
        raw_name = utt.speaker.meta['name'].strip() 
        #print('Error:', case_id, raw_name, last_name)
        return False


    chiefjustice_start_date = chiefjustice2start_date[last_name]

    #decision date
    ddate = caseid2stuff[case_id]['decided_date']
    if ddate == None: return False
    decided_date = datetime.datetime.strptime(ddate, '%b %d, %Y').strftime('%Y-%d-%m')
   
    if decided_date < chiefjustice_start_date:
        #ipdb.set_trace()
        return False
    else: 
        return True

def get_justice_last_name(utt): 
    """
    Returns the last name of the Justice (cased)

    Errors we were getting: 
        Removes 'Jr.' 
    """
    name = utt.speaker.meta['name'].strip()
    name = name.replace(' II', ' ')
    name = name.replace(', Jr.', ' ')
    last_name = name.strip().split(' ')[-1]
    return last_name

def get_corrected_speaker_type(case_id, caseid2stuff, utt):
    """
    Sometimes justices who were previously advocates get
    labeled incorrectly depending on the year
    
    If the case_decide_date < justice_start_date
        the justice was an advocate 
    """ 
    speaker_type = utt.speaker.meta['type']
    
    #sometimes a speaker is neither A nor J and something like <INAUDIBLE>
    if speaker_type not in ["J", "A"]: return None 

    #only need to look at false positives for justices 
    if speaker_type != "J":
        speaker_type == "A" 
        return speaker_type

    justice2start_date = load_justice2start_date()

    #make justice last name
    last_name = get_justice_last_name(utt) 
    
    if justice2start_date.get(last_name) == None:
        raw_name = utt.speaker.meta['name'].strip() 
        print('Error:', case_id, raw_name, last_name)
        return None

    justice_start_date = justice2start_date[last_name]

    #decision date
    ddate = caseid2stuff[case_id]['decided_date']
    if ddate == None: return speaker_type
    decided_date = datetime.datetime.strptime(ddate, '%b %d, %Y').strftime('%Y-%d-%m')

    if decided_date < justice_start_date:
        #ipdb.set_trace()
        return "A"
    else: 
        return "J"

def classify_interruption(utt_text, interruption_symbols=['--', '...']):
    """
    An interruption happens if the interruption symbols are in the last part of the 
    utterance 
    """
    utt_text = utt_text.strip()
    for inter_symbol in interruption_symbols: 
        last = utt_text[-len(inter_symbol):]
        if last == inter_symbol: return True 
    return False 

def load_case_file(): 
    """
    wget https://zissou.infosci.cornell.edu/convokit/datasets/supreme-corpus/cases.jsonl
    """
    caseid2stuff = {}
    for line in open('../raw_data/cases.jsonl', 'r'): 
        dd = json.loads(line)
        case_id = dd['id']
        caseid2stuff[case_id] = dd 
    return caseid2stuff 

def load_justice_ideologies():
    #Doug created the dictionary loaded from: data/justice-ideology.txt manually
    with open('../raw_data/justice-ideology.txt') as f:
        data = f.read()
        return json.loads(data)
        
def load_docket_info():
    import pandas as pd
    return pd.read_csv('../raw_data/scdb_docket.csv',encoding='cp1252')

def is_female_issue(caseid2stuff,df,caseid,advocatename):
    docketid = caseid2stuff[caseid]["scdb_docket_id"]
    issue_num = df.loc[df['docketId'] == docketid]["issue"]
    
    # edge cases 
    if issue_num.empty: return 0 
    if math.isnan(issue_num):
        return 0

    # convert to int 
    issue_num = int(issue_num.iloc[0])
    
    # The issue numbers that are considered "gendered"
    if  (  issue_num == 20130 
        or issue_num == 20140 
        or issue_num == 50020 
        or issue_num == 50010):
        return 1

    return 0 

def get_advocate_ideology(caseid2stuff,df,caseid,advocatename):
    import math
    docketid = caseid2stuff[caseid]["scdb_docket_id"]
    if(len(df.loc[df['docketId'] == docketid]["decisionDirection"]) ==0 or math.isnan(df.loc[df['docketId'] == docketid]["decisionDirection"])):
        decisionDirection = "unknown"
    elif (int(df.loc[df['docketId'] == docketid]["decisionDirection"]) == 1):
        decisionDirection = "conservative"
    elif (int(df.loc[df['docketId'] == docketid]["decisionDirection"]) == 2):
        decisionDirection = "liberal"
    else:
        decisionDirection = "unknown"
    winSide = caseid2stuff[caseid]["win_side"]
    if advocatename in caseid2stuff[caseid]["advocates"]:
        advocateSide = caseid2stuff[caseid]["advocates"][advocatename]["side"]
    else:
        advocateSide = 3
        print(advocatename, ' not found in caseid2stuff dict, assigning unknown. case id: '+caseid)
    
    #advocateSide is 0 for the respondent, 1 for the petitioner, 2 for amicus curiae, 3 for unknown
    #winSide: 1 if the case (in which the session occurred) was decided favorably for the petitioner, 
    # 0 if it wasn’t; 2 if the decision was unclear, and -1 if this information was unavailable
    if (advocateSide == 2 or advocateSide == 3 or winSide == 2 or winSide == -1 or decisionDirection == "unknown"):
        return "unknown"
    elif ((decisionDirection == "conservative" and winSide == advocateSide ) or (decisionDirection == "liberal" and winSide != advocateSide)):
        return "conservative"
    else:
        return "liberal"

def load_config(): 
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    #print("Loading config.yaml")
    assert type(config) == dict 
    return config 
    
def load_justice_gender(): 
    """
    manually coded via: https://en.wikipedia.org/wiki/List_of_justices_of_the_Supreme_Court_of_the_United_States
    """
    justice2gender = {'Anthony M. Kennedy': 'M',
                 'Antonin Scalia': 'M',
                 'Byron R. White': 'M',
                 'David H. Souter': 'M',
                 'Elena Kagan': 'F',
                 'John G. Roberts Jr.': 'M',
                 'John Paul Stevens':'M',
                 'Ruth Bader Ginsburg': 'F',
                 'Samuel A. Alito Jr.':'M',
                 "Sandra Day O'Connor": 'F',
                 'Sonia Sotomayor': 'F',
                 'Stephen G. Breyer': 'M',
                 'Thurgood Marshall': 'M',
                 'William H. Rehnquist': 'M',
                 'Warren E. Burger': 'M',
                 'Neil Gorsuch':'M',
                 'Harry A. Blackmun':'M', 
                 'Brett M. Kavanaugh':'M',
                 'Charles E. Whittaker':'M',
                 'Clarence Thomas': 'M',                     
                 'Earl Warren': 'M',
                 'Hugo L. Black':'M', 
                 'Lewis F. Powell Jr.': 'M',
                 'Potter Stewart': 'M', 
                 'William J. Brennan Jr.': 'M',
                 'William O. Douglas': 'M'}
    return justice2gender

def load_chunks_df(config): 
    """
    Loads the data frame with the chunks 
    (after chunking with create_analyze_chunks)
    """
    num_exclude_adv_first_utt = 0 
    df = []
    for fname in glob.glob(config['chunk_path']+"*"): 
        for line in open(fname, 'r'): 
            dd = json.loads(line)
            if dd['case_year'] < config['start_year']: 
                continue 
            if config['exclude_adv_first_utt']==True and dd['utt_id_first'].split('_')[-1] in ['000', '001']: 
                num_exclude_adv_first_utt += 1
            else:
                df.append(dd)

    df = pd.DataFrame(df) 

    assert df.shape == df.drop_duplicates().shape

    print(f'num_exclude_adv_first_utt={num_exclude_adv_first_utt}')
    print('num chunks = len(df)=', len(df))
    return df 

def load_final_df(config): 
    """
    Loads the df created after first running (1) create_analyze_chunks.py and (2) justice_filter.py 
    """
    df = pd.read_csv(config['final_df_path'])
    print("Loaded final df from ", config['final_df_path'])
    print("Number of rows=", len(df))
    return df 

def create_df_by_just(df_local, treatment_column):
    df_by_just = df_local.groupby(['justice_name', treatment_column]).agg({'utt_id_first': 'count', 'adv_interruption_rate': 'mean'})
    df_by_just = df_by_just.reset_index()
    df_by_just = df_by_just.rename(columns={'utt_id_first': 'num_chunks', 'adv_interruption_rate': 'E[Y]'})
    return df_by_just.copy()

def add_gender_ideo(result_df_reset, df_original):
    # Add back in the gender an ideology 
    df_small = df_original[['justice_name', 'justice_gender', 'justice_ideology']]
    out = result_df_reset.merge(df_small, left_on='justice_name', right_on='justice_name', how='inner').drop_duplicates(subset=['justice_name']).reset_index(drop=True)
    return out 

def calc_theta_ideology(df_local, join_back=True): 
    """
    Calculates theta_ideological alignment for each justice 
    E[Y|Ideology alignment = 1] - E[Y|Ideology alignment = 0]

    join_back == True means to add back the gender and ideology joined back with the df  
    """
    df_by_just = create_df_by_just(df_local.copy(), 'ideology_matches')

    #pivot the table
    grouped_df = df_by_just.groupby(['justice_name', 'ideology_matches']).agg({'E[Y]': 'mean', 'num_chunks': 'sum'}).unstack() #aggregates don't actually matter here  
    
    # Theta
    grouped_df['theta'] = grouped_df['E[Y]'][1] - grouped_df['E[Y]'][0]

    #num chunks 
    grouped_df['total_num_chunks'] = grouped_df['num_chunks'][1] + grouped_df['num_chunks'][0]

    # Flatten index
    result_df = grouped_df[['theta', 'total_num_chunks']]
    result_df_reset = result_df.reset_index()
    result_df_reset.columns = ['justice_name', 'theta', 'total_num_chunks']
    
    if join_back:return add_gender_ideo(result_df_reset.copy(), df_local.copy())
    else: return result_df_reset

def calc_theta_gender(df_local, join_back=True): 
    """
    Calculates theta_gender for each justice 
    E[Y|Advocate gender = F] - E[Y|Advocate gender = M]

    join_back == True means to add back the gender and ideology joined back with the df  
    """
    df_by_just = create_df_by_just(df_local, 'advocate_gender')

    #pivot the table
    grouped_df = df_by_just.groupby(['justice_name', 'advocate_gender']).agg({'E[Y]': 'mean', 'num_chunks': 'sum'}).unstack() #aggregates don't actually matter here  
    
    # Theta
    grouped_df['theta'] = grouped_df['E[Y]']['F'] - grouped_df['E[Y]']['M']

    #num chunks 
    grouped_df['total_num_chunks'] = grouped_df['num_chunks']['F'] + grouped_df['num_chunks']['M']

    # Flatten index
    result_df = grouped_df[['theta', 'total_num_chunks']]
    result_df_reset = result_df.reset_index()
    result_df_reset.columns = ['justice_name', 'theta', 'total_num_chunks']
    
    if join_back:return add_gender_ideo(result_df_reset.copy(), df_local.copy())
    else: return result_df_reset

def calc_ey(df_local, join_back=True):
    """
    Calculate the E[Y] for each justice

    join_back == True means to add back the gender and ideology joined back with the df  
    """
    df_by_just = df_local.groupby(['justice_name']).agg({'utt_id_first': 'count', 'adv_interruption_rate': 'mean'})
    df_by_just = df_by_just.reset_index()
    df_by_just = df_by_just.rename(columns={'utt_id_first': 'num_chunks', 'adv_interruption_rate': 'E[Y]'})
    
    if join_back:return add_gender_ideo(df_by_just.copy(), df_local.copy())
    else: return df_by_just

def plot_confidence_interval(ax, y, mean, stdev, color='blue', horizontal_line_width=0.25, shift=False):
    confidence_interval = 1.96 * stdev
    
    top = y - horizontal_line_width / 2
    left = mean - confidence_interval
    bottom = y + horizontal_line_width / 2
    right = mean + confidence_interval
    
    ax.plot([left, right], [y, y], color=color)
    ax.plot([left, left], [top, bottom], color=color)
    ax.plot([right, right], [top, bottom], color=color)
    ax.plot(mean, y, 'o', color='green', markersize=13)
    
    if shift==True and y==6:
        ax.text(mean-3, y+0.2,
             "{:.2f}".format(np.round(mean, 2))+r'$\pm$'+"{:.2f}".format(np.round(1.96*stdev, 2)),\
             fontsize=30)
    else:
        ax.text(mean-0.7, y+0.2,
             "{:.2f}".format(np.round(mean, 2))+r'$\pm$'+"{:.2f}".format(np.round(1.96*stdev, 2)),\
             fontsize=30)
    
    return mean, confidence_interval

def backchannel_match(utt_text, cues): 
    """
    First, we strip the utt_text of ending punctuation and lowercase 

    Then, if there is an exact match between it and any of the cues, we return True
    otherwise, return False

    Testing 
    >>> cues = load_backchannel_cues()
    >>> assert backchannel_match("Right.", cues) == True
    >>> assert backchannel_match("Right you are I say.", cues) == False
    >>> assert backchannel_match("That's right.", cues) == True
    >>> assert backchannel_match("You don't know if that's right", cues) == False

    """
    #String of only ending punctuation and lower case 
    pattern = r'[^\w\s]+$'# Regular expression pattern to match ending punctuation
    utt_text = re.sub(pattern, '', utt_text).lower()

    for word in cues: 
        if word == utt_text: return True 
    return False 

def load_backchannel_cues(): 
    cues = []
    fname = '../raw_data/backchannel.txt'
    with open(fname, 'r') as r: 
        for line in r: 
            cues.append(line.strip())
    return cues

def get_bootstrap_std(df, num_bootstraps=100): 
    """
    Runs non-parametric bootstrap for E[Y], theta_gender, and theta_ideology
    simultaneously

    1. Subset data frame by justice 
    2. For each justice, sample with replacement same number of chunks 
    
    Calculate new E[Y], theta_gender, and theta_ideology

    Return standard deviations of all bootstraps

    Output: Dictionary 
    - keys are ey, gender, ideology, justices 
    - values are arrays/list wiht the std of the values per justice  
    """
    justices = sorted(df['justice_name'].unique())

    stuff = {
        'ey': [], 
        'gender': [],
        'ideology': [], 
    }

    out = {}
     
    for i in tqdm(range(num_bootstraps)): 
        df_resample = []
        for justice in justices: 
            df_just_only = df[df['justice_name'] == justice]
            sampled_df = df_just_only.sample(n=len(df_just_only), replace=True)
            assert len(sampled_df) == len(df_just_only)
            sampled = sampled_df.to_dict(orient='records') #convert to list of dicts, where each row is now a dict
            df_resample += sampled
        df_resample = pd.DataFrame(df_resample)
        assert len(df_resample) == len(df)

        ey_results = calc_ey(df_resample, join_back=False)
        gender_results = calc_theta_gender(df_resample, join_back=False) 
        idelogy_results = calc_theta_ideology(df_resample, join_back=False) 

        #make sure in the correct justice order 
        np.testing.assert_array_equal(justices, 
                                        ey_results['justice_name'].to_numpy(), 
                                        gender_results['justice_name'].to_numpy(),
                                        idelogy_results['justice_name'].to_numpy())
        
        #then just take the columns 
        stuff['ey'].append(ey_results["E[Y]"].to_list())
        stuff['gender'].append(gender_results["theta"].to_list())
        stuff['ideology'].append(idelogy_results["theta"].to_list())

    # now get the standard deviations 
    for key in stuff.keys(): 
        std = np.std(np.array(stuff[key]), axis=0)
        assert len(std) == len(justices)
        out[key] = std
    
    out['justices'] = justices
    return out 