"""
This file contains functions that designate where chunks start and end and that write metadata for all valid chunks into a dictionary.
Individual functions have more detailed description. 

"""
import os, sys, re
from convokit import Corpus, download
import datetime, argparse
import json
import math
import yaml

import nltk
from nltk import word_tokenize
nltk.download("punkt")

import utils
from advocate_gender import *
from utils import *
import matplotlib.pyplot as plt
import datetime

all_total_backchannel_utts_ignored = 0 

def print_prev_utt_for_chunk(corpus, caseid2stuff, case="2019_17-834"):
    """
    Output: the final utterance of each chunk of conversation

    Example:
    ['25032__0_000', '25032__0_001', '25032__0_003', '25032__0_007', '25032__0_009', '25032__0_026', 
    '25032__0_029', '25032__0_038', '25032__0_056', '25032__0_058', '25032__0_060', '25032__0_064', 
    '25032__0_065', '25032__1_002', '25032__1_028', '25032__1_032', '25032__1_038', '25032__1_042'...

    Rules: 
    An advocate is always first in a chunk of at least two utterances
    If a justice is first, then the chunk has only the one utterance from the justice
    A _valid_ chunk will always have an even number of utterances except:
        -------sometimes a justice or advocate speaks in two consecutive utterances. We would like to keep both,  but this makes our utterance count odd.
        -------(Rare) When a valid chunk is with the chief justice, we have (adv-just-adv-just-...) but recall that the chief justice also introduces the next person. 
        Only in this case, we have that such an utterance ends with the advocate (because that final line belongs to its own chunk). When such a chunk ends 
        with the advocate, given that the chunk started with an advocate, and given that no justice or advocate in the chunk speaks in two consecutive 
        utterances, the chunk will have an odd length.
    """
    arr = []
    curr_case_id = -1
    spkr1 = None #None indicates that we do not know who the speaker is yet
    spkr2 = None
    prev_utt = None
    prev_utt_id = -1
    prev_spkr_name = None
    prev_utt_spkr_label = -1

    all_utt = corpus.get_utterance_ids()
    for i, utt in enumerate(corpus.iter_utterances()):
        case_id = utt.meta['case_id']
        #if we want to get all previous utterances for all case IDs (the argument is case=None) 
        # or if we just want to get the previous utterance for a specific case ID (arg is case="...")
        if (case is not None):
            if (case_id != case):
                continue
        if (curr_case_id != case_id):
            curr_case_id=case_id
            spkr1 = None
            spkr2 = None
            second_prev_utt_id = -1
            prev_utt_id = -1
            prev_spkr_name = None
            prev_utt_spkr_label = -1
        speaker_name = utt.speaker.meta['name'].replace(',', '')
        utt_id = utt.id 
        utt_spkr_label = utt_id.split('_')[2]
        # checking only the first line of text in the conversation (which is probably the chief justice)
        if (prev_utt_spkr_label == -1 and (is_chiefjustice_speaking(case_id,caseid2stuff,utt) == True)):
            arr.append(utt_id)

        # checking if we are changing advocates; in that case, we would like to 
        # designate that the chief justice's line makes up one chunk. 
        # Therefore, we add the previous utterance to mark the end of the chief justice's line, 
        # and the utterance before that to mark the beginning of the chief justice's line (second_prev_utt_id), 
        # if it has not already been added as a previous utterance.
        elif (prev_utt_spkr_label != utt_spkr_label and (prev_utt is not None) and (is_chiefjustice_speaking(case_id,caseid2stuff,prev_utt) == True)):
            spkr1 = speaker_name
            spkr2 = None
            if (second_prev_utt_id not in arr):
                arr.append(second_prev_utt_id)
            if (prev_utt_id not in arr):
                arr.append(prev_utt_id)

        # checking 1. if we don't know the speakers in the current chunk that we're looking at we'll have to initialize who the speakers in this chunk are 
        # and 2. if we do know who the speakers in the current chunk but the speaker of this new utterance is different from the speakers of the current chunk 
        # (in this case, we end the current chunk and start a new chunk). 
        # The piece of code (prev_spkr_name != "John G. Roberts Jr.") is just to avoid adding duplicates.
        elif ((spkr1 is None or spkr2 is None or (spkr1 != speaker_name and spkr2 != speaker_name)) and (prev_utt_spkr_label == utt_spkr_label)):
            if (spkr1 is None):
                if (get_corrected_speaker_type(utt.meta['case_id'], caseid2stuff, utt)=="A"):
                    spkr1 = speaker_name
                if (prev_utt_id not in arr):
                    arr.append(prev_utt_id)
            elif (spkr2 is None):
                spkr2 = speaker_name
            else:
                #if an advocate is speaking
                if (get_corrected_speaker_type(utt.meta['case_id'], caseid2stuff, utt)=="A"):
                    if (prev_utt_id not in arr):
                        arr.append(prev_utt_id)
                    spkr1 = speaker_name
                    spkr2 = None
                #if an advocate was the previous speaker
                elif (prev_utt is not None and get_corrected_speaker_type(prev_utt.meta['case_id'], caseid2stuff, prev_utt)=="A"):
                    spkr1 = prev_spkr_name
                    spkr2 = speaker_name
                    if (second_prev_utt_id not in arr):
                        arr.append(second_prev_utt_id)
                #the advocate is not speaking in this or the previous utterance
                else:
                    spkr1 = None
                    spkr2 = None
                    if (prev_utt_id not in arr):
                        arr.append(prev_utt_id)
        second_prev_utt_id = prev_utt_id
        prev_utt = utt
        prev_utt_id = utt_id
        prev_utt_spkr_label=utt_spkr_label
        prev_spkr_name=speaker_name
    path = config["prev_utt_path"]
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(path)
    with open( config["prev_utt_path"]+case + '.json', 'w') as json_file:
        json.dump(arr, json_file)
    return arr


def analyzechunks(corpus,caseid2stuff, name2gender,caseid2gender,utt_list,seen_advocates,min_num_utts=4,min_tok_adv=20):
    """
    Output: This function writes to a jsonl file metadata for all chunks corresponding to one case.  

    Example:
    {"case_id": "1987_86-594", "case_year": 1987, "justice_name": "Antonin Scalia", "advocate_name": "Laurence E. Gold", "utt_id_first": "18304__1_064", "utt_id_last": "18304__1_077", "advocate_gender": "M", "num_utts": 14, "num_utts_adv": 7, "num_utts_justice": 7, "num_toks_total": 677, "num_toks_adv": 291, "num_toks_justice": 386, "advocate_ideology": "liberal", "justice_ideology": "conservative", "adv_experience": 1, "female_issue": 0, "num_adv_utts_interrupted": 2, "num_justice_utts_interrupted": 1, "adv_interruption_rate": 0.2857142857142857, "justice_interruption_rate": 0.14285714285714285, "num_adv_disfl": 5, "num_justice_disfl": 1, "num_adv_toks_in_utts_interrupted": 10, "num_justice_toks_in_utts_interrupted": 152}    {"case_id": "2015_13-1067", "case_year": 2015, "justice_name": "Elena Kagan", "advocate_name": "Juan C. Basombrio", "utt_id_first": "23997__0_007", "utt_id_last": "23997__0_010", "advocate_gender": "M", "num_utts": 4, "num_utts_adv": 2, "num_utts_justice": 2, "num_toks_total": 345, "num_toks_adv": 67, "num_toks_justice": 278, "advocate_ideology": "conservative", "justice_ideology": "liberal", "num_adv_utts_interrupted": 1, "interruption_rate": 0.5, "num_adv_disfl": 0, "num_justice_disfl": 8}
    {"case_id": "1986_85-1835", "case_year": 1986, "justice_name": "Antonin Scalia", "advocate_name": "Arthur Lewis", "utt_id_first": "19147__1_064", "utt_id_last": "19147__1_075", "advocate_gender": "M", "num_utts": 12, "num_utts_adv": 6, "num_utts_justice": 6, "num_toks_total": 725, "num_toks_adv": 425, "num_toks_justice": 300, "advocate_ideology": "liberal", "justice_ideology": "conservative", "adv_experience": 0, "female_issue": 0, "num_adv_utts_interrupted": 1, "num_justice_utts_interrupted": 0, "adv_interruption_rate": 0.16666666666666666, "justice_interruption_rate": 0.0, "num_adv_disfl": 1, "num_justice_disfl": 1, "num_adv_toks_in_utts_interrupted": 58, "num_justice_toks_in_utts_interrupted": 0}
    ...

    """
    # Load metadata 
    justice_ideologies_dict = load_justice_ideologies()
    name2gender = create_load_lookupname2gender() 
    df = load_docket_info() 
    cues = load_backchannel_cues()
    total_backchannel_utts_ignored = 0 

    dict_list = []
    if (len(utt_list)==0):
        return seen_advocates
    case = corpus.get_utterance(utt_list[0]).meta['case_id']
    
    config = load_config()
    path = config['chunk_path']
    if not os.path.exists(path):os.makedirs(path)
    
    f = open(config['chunk_path']+case + '.jsonl', 'w') 
    if (-1 in utt_list):
        utt_list.remove(-1)
    
    all_utt = corpus.get_utterance_ids()
    prev_utt_p1 = -1
    prev_utt_p3 = -1
    prev_utt_id = -1
    advocates_in_this_case = []
    for utt_id in utt_list:
        utt_p1 = utt_id.split('_')[0]
        utt_p3 = utt_id.split('_')[3]
        #if the chunk has only one speaker 
        if (prev_utt_p1 == -1 or prev_utt_p3 == -1 or prev_utt_p1 != utt_p1 or int(utt_p3) == int(prev_utt_p3)+1):
            x=1 #do nothing
        #if the chunk has more than one speaker
        elif (prev_utt_id != -1): 
            
            
            #print speakers in the chunk
            #print the first two speakers in the chunk, which are:
            # the immediate next speaker of the utterance after prev_utt: call this prev_next_utt
            # the immediate next speaker of the utterance after prev_next_utt: call this prev_next2_utt
            prev_next_utt_id = all_utt[all_utt.index(prev_utt_id)+1]
            prev_next_utt = corpus.get_utterance(prev_next_utt_id)
            spkr1 = prev_next_utt.speaker.meta['name'].replace(',', '')
            prev_next2_utt_id = all_utt[all_utt.index(prev_utt_id)+2]
            prev_next2_utt = corpus.get_utterance(prev_next2_utt_id)
            spkr2 = prev_next2_utt.speaker.meta['name'].replace(',', '')            
            
            #determine if the chunk is "valid"
            if ((int(utt_p3) >= int(prev_utt_p3)+min_num_utts) 
                    or ((int(prev_next_utt_id.split('_')[3])==0) 
                    and (int(utt_p3) >=min_num_utts-1))):
                
                num_utt = int(utt_p3) - int(prev_utt_p3)

                if (int(prev_next_utt_id.split('_')[3])==0):
                    num_utt =  int(utt_p3) + 1
                spkr1type = get_corrected_speaker_type(prev_next_utt.meta['case_id'], caseid2stuff, prev_next_utt)
                spkr2type = get_corrected_speaker_type(prev_next2_utt.meta['case_id'], caseid2stuff, prev_next2_utt)
                
                if ((spkr1type == "J" and spkr2type == "A") or (spkr1type == "A" and spkr2type == "J")):
                    #valid chunk
                    utt =  corpus.get_utterance(utt_id)
                    caseid = utt.meta['case_id']
                    uttidlast = utt_id
                    uttidfirst = prev_next_utt_id
                    caseyear = int(caseid.split('_')[0])
                    if (spkr1type == "J"):
                        justicename = spkr1
                        advocatename = spkr2
                    else:
                        justicename = spkr2
                        advocatename = spkr1

                    #Justice last name check 
                    if (justicename.split()[len(justicename.split())-1] != "Jr."):
                        justicelastname = justicename.split()[len(justicename.split())-1]
                    else:
                        justicelastname = justicename.split()[len(justicename.split())-2]

                    # Get justice ideology 
                    if justicelastname in justice_ideologies_dict:    
                        justice_ideology = justice_ideologies_dict[justicelastname]
                    else:
                        justice_ideology = "unknown"

                    # Advocate ideology 
                    advocate_ideology = get_advocate_ideology(caseid2stuff,df,caseid,advocatename)
                    female_issue = is_female_issue(caseid2stuff,df,caseid,advocatename)
                    if (advocatename in caseid2gender[caseid] and (caseid2gender[caseid][advocatename]=="M" or caseid2gender[caseid][advocatename]=="F")):
                        gender = caseid2gender[caseid][advocatename]
                    else: 
                        gender = get_speaker_gender_dictionary(advocatename, name2gender)
                    if (advocatename in seen_advocates):
                        adv_experience_bin = 1
                        adv_experience_int = seen_advocates[advocatename]
                    else:
                        adv_experience_bin = 0
                        adv_experience_int = 0
                    if advocatename not in advocates_in_this_case:
                        advocates_in_this_case.append(advocatename)
                    
                    num_utts_adv = 0
                    num_utts_justice = 0
                    num_toks_total = 0
                    num_toks_adv = 0
                    num_toks_justice = 0
                    num_adv_utts_interrupted = 0
                    num_justice_utts_interrupted = 0
                    num_adv_toks_in_utts_interrupted = 0
                    num_justice_toks_in_utts_interrupted = 0
                    num_adv_disfl = 0 
                    num_justice_disfl = 0
                    
                    for u in range(1, num_utt + 1):
                        utter = corpus.get_utterance(all_utt[all_utt.index(prev_utt_id)+u])
                        text = utter.text.strip()

                        # Check backchannel cues (if applicable)
                        if config["exclude_backchannel"] == True: 
                            has_backchannel = backchannel_match(text, cues)
                            if has_backchannel == True: 
                                total_backchannel_utts_ignored  += 1
                                continue

                        # Otherwise, continue on 
                        tokenized_text = word_tokenize(text)
                        num_toks = len(tokenized_text)
                        num_toks_total = num_toks_total + num_toks

                        if (get_corrected_speaker_type(utter.meta['case_id'],caseid2stuff,utter)=="A"):
                            num_utts_adv = num_utts_adv + 1
                            num_toks_adv = num_toks_adv + num_toks
                            if (classify_interruption(text)==True):
                                num_adv_utts_interrupted = num_adv_utts_interrupted + 1
                                num_adv_toks_in_utts_interrupted = num_adv_toks_in_utts_interrupted+num_toks
                            num_adv_disfl = num_adv_disfl + one_utt_rule_speech_disfluency(tokenized_text)
                        else:
                            num_utts_justice = num_utts_justice + 1 
                            num_toks_justice =  num_toks_justice + num_toks
                            if (classify_interruption(text)==True):
                                num_justice_utts_interrupted = num_justice_utts_interrupted + 1
                                num_justice_toks_in_utts_interrupted = num_justice_toks_in_utts_interrupted+num_toks
                            num_justice_disfl = num_justice_disfl + one_utt_rule_speech_disfluency(tokenized_text)
                    
                    # Could end up with invalid num utterances if all backchannels
                    if num_utts_adv < 2 or num_utts_justice <2 : continue 
                    
                    adv_interruption_rate = num_adv_utts_interrupted / num_utts_adv
                    justice_interruption_rate = num_justice_utts_interrupted / num_utts_justice
                    if (num_toks_adv >= min_tok_adv):
                        dic = dict(case_id=caseid,case_year=caseyear,justice_name=justicename,advocate_name=advocatename,utt_id_first=uttidfirst,utt_id_last=uttidlast,advocate_gender=gender,
                        num_utts=num_utt,num_utts_adv= num_utts_adv,num_utts_justice=num_utts_justice,num_toks_total=num_toks_total,num_toks_adv=num_toks_adv,num_toks_justice=num_toks_justice,advocate_ideology=advocate_ideology,
                        justice_ideology=justice_ideology,adv_experience_int=adv_experience_int,adv_experience_bin=adv_experience_bin,female_issue=female_issue,num_adv_utts_interrupted=num_adv_utts_interrupted,num_justice_utts_interrupted=num_justice_utts_interrupted,adv_interruption_rate=adv_interruption_rate,
                        justice_interruption_rate=justice_interruption_rate,num_adv_disfl=num_adv_disfl,num_justice_disfl=num_justice_disfl, num_adv_toks_in_utts_interrupted=num_adv_toks_in_utts_interrupted,num_justice_toks_in_utts_interrupted=num_justice_toks_in_utts_interrupted)
                        dict_list.append(dic)
                        json.dump(dic, f) 
                        f.write('\n')
        prev_utt_id = utt_id
        prev_utt_p1 = utt_p1
        prev_utt_p3 = utt_p3

    # Advocate experience piece 
    for advs in advocates_in_this_case:
        if advs not in seen_advocates:
            seen_advocates[advs] = 1
        else:
            seen_advocates[advs] = seen_advocates[advs] + 1
    
    #Print some stuff 
    if config["exclude_backchannel"] == True:
        #print("Total backchannel utts ignored = ", total_backchannel_utts_ignored)
        global all_total_backchannel_utts_ignored 
        all_total_backchannel_utts_ignored += total_backchannel_utts_ignored

    return seen_advocates

def analyzechunks1year(corpus1,caseid2stuff,name2gender,caseid2gender,seen_advocates):
    """
    Output: This function generates a jsonl file for each case in a year, where each jsonl file contains metadata for all chunks corresponding to the case corresponding to it.
    """

    #iterates through the list of conversations in a year; sorts conversations in order of argument date
    df = load_docket_info()
    conv_date_list = []
    for i, conv in enumerate(corpus1.iter_conversations()):
        utt_ids = conv.get_utterance_ids()
        utt = conv.get_utterance(utt_ids[0])
        case_id = utt.meta["case_id"]
        docketid = caseid2stuff[case_id]["scdb_docket_id"]
        date_arg = df.loc[df['docketId'] == docketid]["dateArgument"]
        if (len(date_arg) == 0 or pd.isna(date_arg.item())):
            date_arg = datetime.date(2020,10,5)
            print("no argument date available")
        else:
            date_arg = datetime.datetime.strptime(word_tokenize(str(date_arg))[1],"%m/%d/%Y").date()
        conv_date_list.append([conv,date_arg])
    conv_date_list = sorted(conv_date_list, key = lambda x: x[1])
    #iterates through the list of conversations in a year, ordered by argument date
    for tup in conv_date_list: 
        conv = tup[0]
        utt_ids = conv.get_utterance_ids()
        utt = conv.get_utterance(utt_ids[0])
        case_id = utt.meta["case_id"]
        utt_list = print_prev_utt_for_chunk(corpus1, caseid2stuff, case=case_id)
        seen_advocates = analyzechunks(corpus1,caseid2stuff,name2gender,caseid2gender,utt_list,seen_advocates)
    return seen_advocates

#prints metadata over all years
def metadata_all_years(start=2019,end=2020):
    """
    Output: This function generates a jsonl file for each case in a year over a period of many years, where each jsonl file contains metadata for all chunks corresponding to the case corresponding to it.
    """
    #iterate through all years
    seen_advocates = {}
    for year in range(start,end,1):
        corpus1 = Corpus(filename=download("supreme-"+str(year)))
        name2gender = create_load_lookupname2gender() 
        caseid2gender = parse_gender(corpus1, name2gender, verbose=False, start_year=1980)
        caseid2stuff = utils.load_case_file()
        seen_advocates = analyzechunks1year(corpus1,caseid2stuff,name2gender,caseid2gender,seen_advocates)

        if config["exclude_backchannel"] == True:
            print("total backchannel utterances ignored across all cases =", all_total_backchannel_utts_ignored)
      

if __name__ == '__main__':
    all_total_backchannel_utts_ignored = 0 
    if not os.path.exists("data/"): os.makedirs("data/")

    config = load_config()

    #the start year is inclusive; the end year is not inclusive
    metadata_all_years(start=config["start_year"], 
                       end=config["end_year"])
    
    if config["exclude_backchannel"] == True:
        print("ALL CASES, ALL YEARS, total backchannel utterances ignored=", all_total_backchannel_utts_ignored)
    print("DONE CHUNKING")