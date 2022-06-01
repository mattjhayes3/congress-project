import os
import json
import pandas as pd
import re
import numpy as np

escaped_dot = '__dot__'

embeddings_index = {}
with open('../../glove.6B/glove.6B.50d.txt') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        # if word == ".":
        #     print("found period")
        # if not word.isalpha():
        #     print(f"non-alpha '{word}'")
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

oov_counts = dict()

corrections = { "thle": "the",
                "shouldnt": "shouldn't",
                "foll ws": "follows",
                r"Com\. mittee": "Committee",
                "gentlemans": "gentleman's",
                "oclock": "o-clock",
                "lowincome": "low-income",
                "shortterm": "short-term",
                "longterm": "long-term",
                "onehalf": "one-half",
                "onethird": "one-third",
                "twothirds": "two-thirds",
                r"(\d{1,2})year": r"\1-year",
                "DavisBacon": "Davis-Bacon",
                "unanimousconsent": "unanimous-consent",
                "([S|s])enator(s?)elect": r"\1enator\2-elect",
                "([R|r])epresentative(s?)elect": r"\1epresentative\2-elect",
                "RECiRD": "RECORD",
                "costofliving": "cost-of-living",
                "farreaching": "far-reaching",
                "industrys": "industry's",
                "([H|h])ousepassed": r"\1ouse-paseed",
                "NASAs": "NASA's",
                "taxexcept": "tax-except",
                "daytoday": "day-to-day",
                "secretarys": "secretary's",
                "ClintonGore": "Clinton-Gore",
                "AfricanAmerican": "African-American",
                "onbudget": "on-budget",
                "acrosstheboard": "across-the-board",
                "non([sS])ocial": r"non-\1ocial",
                "seoconddegree": "second-degree",
                "middleincome": "middle-income",
                "costeffective": "cost-effective",
                "muchneeded": "much-needed",
                "([Gg])ovemment": r"\1overnment",
                "twentyfive": "twenty-five",
                "agencys": "agency's",
                "hardearned": "hard-earned",
                "selfemployed": "self-employed",
                "familys": "family's",
                "communitybased": "community-based",
                "werent": "weren't",
                "lawabiding": "law-abiding",
                "highquality": "high-quality",
}
corrections = {re.compile(f"\\b{k}\\b") :v for k,v in corrections.items()}

abbreviations_list = ['U.S.', 'U.S.A.', 'Mr.', 'Mrs.', 'Ms.', 'a.m.', 'p.m.', 'A.M.', 'P.M.', 'Hon.', 'Ho.', 'esq.', 'i.e.', 'e.g.', 'Stat.', 'c.']
abbreviations = {}
for a in abbreviations_list:
    escaped = a.replace(".", "\\.")
    updated = a.replace('.', escaped_dot)
    abbreviations[re.compile(f'(\\b){escaped}(\\W)')] = f"\\1{updated}\\2"
four_cap_letter_abbreviation = re.compile(r'(\b)([A-Z])\.([A-Z])\.([A-Z])\.([A-Z])\.(\W)')
three_cap_letter_abbreviation = re.compile(r'(\b)([A-Z])\.([A-Z])\.([A-Z])\.(\W)')
two_cap_letter_abbreviation = re.compile(r'(\b)([A-Z])\.([A-Z])\.(\W)')
one_cap_letter_abbreviation = re.compile(r'(\b)([A-Z])\.(\W)')
number_with_dot = re.compile(r'(\d)\.( \d)')
elipsis = re.compile(r'\.\.\.')

def correctText(speech_text):
    orig = speech_text
    # find the periods, if a word after period is lowercase, replace period with comma
    # remember though, abbreviations als,o have periods. Get rid of some
    speech_text = re.sub(elipsis, f"{escaped_dot}{escaped_dot}{escaped_dot}", speech_text)
    for abbrevation, rewrite in abbreviations.items():
        # escaped = abbrevation.replace('.', '\\.')
        # updated = abbrevation.replace('.', period_replacement)
        # regex = '(\\b)' + escaped + '(\\W)'
        # print(f'regex is "{regex}"')
        # speech_text = re.sub(regex, f"\\1{updated}\\2", speech_text)
        speech_text = re.sub(abbrevation, rewrite, speech_text)
    
    speech_text = re.sub(four_cap_letter_abbreviation,
                         f"\\1\\2{escaped_dot}\\3{escaped_dot}\\4{escaped_dot}\\5{escaped_dot}\\6", speech_text)
    speech_text = re.sub(three_cap_letter_abbreviation,
                         f"\\1\\2{escaped_dot}\\3{escaped_dot}\\4{escaped_dot}\\5", speech_text)
    speech_text = re.sub(two_cap_letter_abbreviation,
                         f"\\1\\2{escaped_dot}\\3{escaped_dot}\\4", speech_text)
    speech_text = re.sub(one_cap_letter_abbreviation,
                         f"\\1\\2{escaped_dot}\\3", speech_text)
    # speech_text = re.sub(r'(\b)([A-Z])\.([A-Z])\.\.(\W)',
    #                      f"\\1\\2{period_replacement}\\3{period_replacement} , \\4", speech_text)
    speech_text = re.sub(number_with_dot, r'\1,\2', speech_text)
    for original, new in corrections.items():
        speech_text = re.sub(original, new, speech_text)
    # if " thle " in speech_text:
    #     print(speech_text.replace(" thle ", " ****thle**** "))
    # then, remaining periods should be mostly the end of sentence info, or those replacing comma's
    period_separated = speech_text.split('.')

    if len(period_separated) < 2:  # to make sure there are at least 2 sentences, or comma separated stuff
        return ''  # not enough sentences to begin with, return just an empty string
    new_text = period_separated[0]

    for piece_index in range(1, len(period_separated)):
        # to check whether the first word of the line following . is starting with uppercase
        first_alpha = re.search("[a-zA-Z]", period_separated[piece_index])
        if len(period_separated[piece_index].strip())>0 and (not first_alpha or first_alpha.group(0).islower()):
            new_text += ' , ' + period_separated[piece_index]
        else:
            new_text += ' . ' + period_separated[piece_index]

    new_text = new_text.replace(escaped_dot, '.').replace(
        '\n', '').replace('  ', ' ')

    # if "today I am introducing a bill to define the affirmative defense of insanity and to establish" in speech_text:
    #     print(f"speech_text: {speech_text}\n\n new: {new_text}")
    words = new_text.split()
    for (i, word) in enumerate(words):
        w = word.lower().strip('?$"[]()-:,')
        if len(w)>0 and w not in embeddings_index:
            if not w in oov_counts:
                oov_counts[w] = []
            oov_counts[w].append(" ".join(words[max(0, i-5): min(i+5, len(words)-1)]))
    return new_text

# check the unsuable and irrelevant texts to eliminate them


def shouldExcludeText(speech_text):

    # eliminate few-sentence long texts
    period_separated = speech_text.split(' . ')

    if len(period_separated) < 3:  # remove those speeches with less than 5 sentences long
        return True
    elif 'take the opportunity to express my appreciation to' in speech_text and 'intern' in speech_text:
        return True  # these speeches are intern appreciation, delete them
    else:
        return False  # rest includes chat about ideologies, bills, complaints, honoring death ppl
    # initially I was going to delete the honorings to death or congratulating people, but I've read some and
    # realized these honorings might be party/ideology related. For example, republicans are honoring those deads that have been
    # church members, and they praise them by their religion. Democrats, on the other hand, congratulating research center openings etc.

# ------------------------------------------------------------------------------------------------------------
# read the speech data, indexed by the speech id's


# selected_congresses = ['097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '114', '113', '112', '111', '110', '109', '108', '107']
# selected_congresses = ['100', '106', '114', '112', '109']
# selected_congresses = ['103']
selected_congresses = [  106, 109, 112, 114]#range(100, )100,103,
# selected_congresses = range(43, 115)

for selected_congress in selected_congresses:
    oov_counts = dict()
    root = "../../congress_data/hein-daily/" if selected_congress >= 97 else "../../congress_data/hein-bound/"
    print(f"processing congress {selected_congress}")

    fmt_congress = "%03d" % selected_congress
    os.makedirs('../../processed_data2/Senate_' + fmt_congress, exist_ok=True)
    os.makedirs('../../processed_data2/House_' + fmt_congress, exist_ok=True)

    speech_path = f"{root}/speeches_{fmt_congress}.txt"
    # collect the speeches of this congress here, associated with it's unique speech id
    # includes house, senate speeches, all.
    speech_dict = {}
    thrown_away_ids = set()

    print("Reading speeches...")
    with open(speech_path, 'r', errors='replace') as speech_fo:
        lines = speech_fo.read().splitlines()
    c = 0
    for line in lines[1:]: # skip the header
        if c % int(len(lines)/20) == 0:
            print(f"{c/len(lines)}%")
        split_line = line.split('|', 1)
        speech_id = split_line[0]
        original_text = split_line[1]

        # here, preprocess (or rather fix) the text's problems, especially create the correct punctuations
        speech_text = correctText(original_text)

        # add this text to the dictionary only if it is long and relevant enough
        if not shouldExcludeText(speech_text):
            speech_dict[int(speech_id)] = speech_text
            if c in [1, 10, 100, 1000, 2000]:
                print(
                    f"adding {speech_id}: {speech_text}\n from {original_text}")
        else:
            thrown_away_ids.add(int(speech_id))
            if c in [1, 10, 100, 1000, 2000]:
                print(f"throwing {speech_id}: {speech_text}")
        c += 1
    # speech_fo.close()
    # print(speech_dict)

    date_fo = open(root + 'descr_' + fmt_congress +
                   '.txt', 'r', errors='replace')
    date_dict = {}
    c = 0
    for line in date_fo:
        if c > 0:
            speech_id = line.split('|')[0]
            date_ = line.split('|')[2]

            date_dict[int(speech_id)] = date_
        c += 1
    date_fo.close()
    # print(date_dict)

    # read the metadata
    print("Joining metadata...")
    df = pd.read_csv(f"{root}{fmt_congress}_SpeakerMap.txt", sep='|')
    for index in df.index:
        row = df.loc[index].to_dict()
        name = f"{row['firstname']} {row['lastname']}"
        row['name'] = name
        if row['speech_id'] not in speech_dict.keys():
            if row['speech_id'] not in thrown_away_ids:
                print(f"Unexpected speech ID {row['speech_id']}")
        else:
            filenamew = f"{row['party'].lower()}_{name.replace(' ', '_').lower()}_{date_dict[row['speech_id']]}_{row['speech_id']}.json"
            row['speech'] = speech_dict[row['speech_id']]
            row['congress'] = selected_congress
            row['date'] = date_dict[row['speech_id']]

            # start writing up what you read here
            if row['chamber'] == 'H':
                # print('writing house ' + filenamew)
                with open('../../processed_data2/House_' + fmt_congress + '/' + filenamew, 'w') as fo_w:
                    fo_w.write(json.dumps(row))
            elif row['chamber'] == 'S':
                # print('writing senate ' + filenamew)
                with open('../../processed_data2/Senate_' + fmt_congress + '/' + filenamew, 'w') as fo_w:
                    fo_w.write(json.dumps(row))
            else:
                print(f"Unexpected chamber {row['chamber']}")
    oov_list = sorted(oov_counts.items(),
                      key=lambda kv: len(kv[1]), reverse=True)
    print(f"top missing words:")
    for i in range(50):
        print(
            f"\t'{oov_list[i][0]}' ({len(oov_list[i][1])}), examples: {[x for x in oov_list[i][1][:5]]}")
