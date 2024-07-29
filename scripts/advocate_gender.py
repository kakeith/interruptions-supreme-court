"""
This script will get the advocates' genders for every case 

Rule-based process: 
    - Get the names of the advocates 
    - Prior to the advocate's first utterance, the Chief Justice will 
    introduce them as "Ms." or "Mr.". We extract this as the gender of the advocate
    - If he doesn't introduce them this way, we look up the advocates first name in a gender dictionary
"""
import json, os
from convokit import Corpus, download
from collections import defaultdict, Counter
import pandas as pd

from utils import *

import nltk
from nltk import word_tokenize
nltk.download("punkt")


def extract_last_gender_title_mention(tokenized_text):
    # go thru in reversed order
    title2gender = {"Mr.": "M", "Ms.": "F"}
    for tok in reversed(tokenized_text):
        if tok in ["Mr.", "Ms."]:
            return title2gender[tok]
    return None


def create_load_lookupname2gender():
    """
    Looks up in a gender in a name mapping, first names map deterministically to a binary gender

    Data from:
        https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YPRQH8#
    """
    fname = "data/name2gender.json"
    if not os.path.exists(fname):
        gender_df = pd.read_csv(
            "../raw_data/wgnd_ctry.csv"
        )  # gendered name dict, https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YPRQH8#
        name2gender = {}
        codes = set()
        for i, row in gender_df.iterrows():
            code = str(row["code"])
            if code != "US":
                continue
            name = str(row["name"]).lower()
            gender = row["gender"]
            name2gender[name] = gender
            codes.add(code)
        print("created name2gender")

        with open(fname, "w") as w:
            json.dump(name2gender, w)
        print("wrote to ", fname)

    else:
        with open(fname, "r") as r:
            name2gender = json.load(r)
            #print("loaded ", fname)

    return name2gender


def get_speaker_gender_dictionary(name, name2gender):
    """
    Get speaker gender based on dictionary of first/last names
    """
    first_name = parse_first_name(name)
    if name2gender.get(first_name) == None:
        return None
    gender = name2gender[first_name]
    if gender in ["M", "F"]:
        return gender
    else:
        return None


def parse_gender(corpus, name2gender, verbose=False, start_year=1980):
    """
    Rule-based process: zx
        - Get the names of the advocates
        - Prior to the advocate's first utterance, the Chief Justice will
        introduce them as "Ms." or "Mr.". We extract this as the gender of the advocate
        - If he doesn't introduce them this way, we look up the advocates first name in a gender dictionary

    Example:
        2019_18-877
        24929__0_000
        We'll hear argument next in Case 18-877, Allen versus Cooper. Mr. Shaffer.
    """
    prev_section = None
    prev_gender_mention = None
    prev_text = None

    caseid2genders = defaultdict(dict)

    for i, utt in enumerate(corpus.iter_utterances()):
        # check to make sure case is in the year we want
        case_id = utt.meta["case_id"]
        year = int(case_id.split("_")[0])
        if year < start_year:
            continue

        if verbose:
            print(case_id)

        # e.g. utt_id=24834__0_00 is section=0
        # and utt_id = 24834__1_018 is section=1
        section = utt.id.split("__")[-1].split("_")[0]

        text = utt.text.strip()
        tokenized_text = [t for t in word_tokenize(text)]
        gender_ment = extract_last_gender_title_mention(tokenized_text)

        # we get the chief justice at the very first utterance
        # OR when a new advocate comes in
        if utt.id.split("__")[-1] == "0_001" or section != prev_section:
            if utt.speaker.meta["type"] != "A":
                if verbose:
                    print(utt.id, utt.speaker.meta["type"], utt.speaker.meta["name"])
            else:
                adv_name = utt.speaker.meta["name"]
                adv_name = adv_name.replace(",", "")  # need to do this for the Jr.s in the csv and to be consistent
                gender = prev_gender_mention

                # Second step: if we don't have a "Mr." or "Mrs." look up the first name in a dictionary
                # Thank you, counsel. General Wall?
                if gender == None:
                    gender = get_speaker_gender_dictionary(adv_name, name2gender)

                if caseid2genders[case_id].get(adv_name) == None:
                    caseid2genders[case_id][adv_name] = gender

                if gender == None:
                    if verbose:
                        print(adv_name, prev_gender_mention, prev_text)

        # reset for next utt
        prev_section = section
        prev_gender_mention = gender_ment
        prev_text = utt.text

    fout = "data/caseid2genders.json"
    with open(fout, "w") as w:
        json.dump(caseid2genders, w)
    print("wrote to {0} cases ->".format(len(caseid2genders)), fout)
    return caseid2genders


if __name__ == "__main__":
    start_year = 1980
    name2gender = create_load_lookupname2gender()
    corpus = Corpus(filename=download("supreme-corpus"))
    parse_gender(corpus, name2gender, verbose=True, start_year=start_year)