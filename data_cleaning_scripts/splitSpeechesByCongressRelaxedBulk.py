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

corrections = { "(thle|tihe)": "the",
                "(shouldn|weren|hadn)t": r"\1't",
                "foll ws": "follows",
                r"Com\. mittee": "Committee",
                "([Cc])(onmittee|oinmittee)": r"\1ommittee",
                "oclock": "o-clock",
                "publie": "public",
                'wlich': 'which',
                "(amendnent|amendmeat)": "amendment",
                "uder": "under",
                "lumpsum": "lump-sum",
                "warprofits": "war-profits",
                "homeloan": "home-loan",
                "(income|sales)tax": r"\1-tax",
                "gentlenan": "gentleman",
                "selfrespect": "self-respect",
                "(high|low)level": r"\1-level",
                "presentday": "present-day",
                "ablebodied": "able-bodied",
                "lowincome": "low-income",
                "(long|short)term": r"\1-term",
                "one(half|third)": r"one-\1",
                "legaltender": "legal-tender",
                "twothirds": "two-thirds",
                "onefifth": "one-fifth",
                "threefifths": "three-fifths",
                r"(\d{1,2})(year|minute)": r"\1-\2",
                "DavisBacon": "Davis-Bacon",
                "unanimousconsent": "unanimous-consent",
                "([S|s])enator(s?)elect": r"\1enator\2-elect",
                "([R|r])epresentative(s?)elect": r"\1epresentative\2-elect",
                "RECiRD": "RECORD",
                "SSIONAL": "CONGRESSIONAL",
                "(they|you)(ll)": r"\1'\2",
                "costofliving": "cost-of-living",
                "farreaching": "far-reaching",
                "(community|faith)based": r"\1-based",
                "([H|h])ousepassed": r"\1ouse-paseed",
                "taxexcept": "tax-except",
                "minimumwage": "minimum-wage",
                "partialbirth": "partial-birth",
                "theyd": "they'd",
                "daytoday": "day-to-day",
                "upordown": "up-or-down",
                "ClintonGore": "Clinton-Gore",
                "AfricanAmerican": "African-American",
                "onbudget": "on-budget",
                "Iouse of Representaatives": "House of Representaatives",
                "samesex": "same-sex",
                "one(fourth|tenth)": r"one-\1",
                "threefourths": "three-fourths",
                "publicland": "public-land",
                "civilservice": "civil-service",
                "(high|at)risk": r"\1-risk",
                "acrosstheboard": "across-the-board",
                "non([sS])ocial": r"non-\1ocial",
                "(first|second|third)(class|degree)": r"\1-\2",
                "(middle|moderate)income": r"\1-income",
                "costeffective": "cost-effective",
                "muchneeded": "much-needed",
                "freetrade": "free-trade",
                "froin": "from",
                "progrowth": "pro-growth",
                "deficitneutral": "deficit-neutral",
                "redtape": "red-tape",
                "exportimport": "export-import",
                "([Gg])ovemment": r"\1overnment",
                "twenty(five|four)": r"twenty-\1",
                "([Ss])eventyfive": r"\1eventy-five",
                "(congresss|congross|Congross)": "congress",
                "(NASA|Ohio|Alaska|Pennsylvania|univeristy|University|gentleman|agency|family|industry|secretary|majority|minority|country|county|County|community)s": r"\1's",
                "onesizefitsall": "one-size-fits-all",
                "hardearned": "hard-earned",
                "morninghour": "morning-hour",
                "awardwinning": "award-winning",
                "smallbusiness": "small-business",
                "evidencebased": "evidence-based",
                "worldclass": "world-class",
                "self(employed|determination)": r"self-\1",
                "lawabiding": "law-abiding",
                "welldeserved": "well-deserved",
                "goodpaying": "good-paying",
                "forprofit": "for-profit",
                "job(killing|creating)": r"job\1",
                "outofcontrol": "out-of-control",
                "(high|low)quality": r"\1-quality",
}
corrections = {re.compile(f"\\b{k}\\b") :v for k,v in corrections.items()}

abbreviations = {
    '(Mr|Mrs|Ms|Hon|Ho|esq|Stat|c)\\.': f"\\1{escaped_dot}",
    r'(a|p)\.m\.': f"\\1{escaped_dot}m{escaped_dot}",
    r'i\.e\.': f'i{escaped_dot}e{escaped_dot}', 
    r'e\.g\.': f'e{escaped_dot}g{escaped_dot}',
}
abbreviations = {re.compile(k):v for k,v in abbreviations.items()}
# abbreviations_list = ['Mr.', 'Mrs.', 'Ms.', 'a.m.', 'p.m.','Hon.', 'Ho.', 'esq.', 'i.e.', 'e.g.', 'Stat.', 'c.',]
# abbreviations = {}  # 'U.S.', 'U.S.A.',  'A.M.', 'P.M.', 
# for a in abbreviations_list:
#     escaped = a.replace(".", "\\.")
#     updated = a.replace('.', escaped_dot)
#     abbreviations[re.compile(f'(\\b){escaped}(\\W)')] = f"\\1{updated}\\2"

four_cap_letter_abbreviation = re.compile(r'(\b)([A-Z])\.([A-Z])\.([A-Z])\.([A-Z])\.(\W)')
three_cap_letter_abbreviation = re.compile(r'(\b)([A-Z])\.([A-Z])\.([A-Z])\.(\W)')
two_cap_letter_abbreviation = re.compile(r'(\b)([A-Z])\.([A-Z])\.(\W)')
one_cap_letter_abbreviation = re.compile(r'(\b)([A-Z])\.(\W)')
number_with_dot = re.compile(r'(\d)\.( \d)')
elipsis = re.compile(r'\.\.\.')
alpha = re.compile("[a-zA-Z]")

def doBulkReplacements(speech_text):
    # original_text = speech_text
    speech_text = re.sub(number_with_dot, r'\1,\2', speech_text)
    z = 0
    for original, new in corrections.items():
        speech_text = re.sub(original, new, speech_text)
        z += 1
        print("correction", z)
    speech_text = re.sub(elipsis, f"{escaped_dot}{escaped_dot}{escaped_dot}", speech_text)
    print("start abbreviations")
    for abbrevation, rewrite in abbreviations.items():
        # escaped = abbrevation.replace('.', '\\.')
        # updated = abbrevation.replace('.', period_replacement)
        # regex = '(\\b)' + escaped + '(\\W)'
        # print(f'regex is "{regex}"')
        # speech_text = re.sub(regex, f"\\1{updated}\\2", speech_text)
        speech_text = re.sub(abbrevation, rewrite, speech_text)
    print("abbreviations done")
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

    # idx = original_text.index(' and followed by a "bullet" symbol. i.e.. 0. 4. Return of manuscript.-W')
    # print(f"found!!  {speech_text}")
    return speech_text

def correctText(speech_text):
    orig = speech_text
    # find the periods, if a word after period is lowercase, replace period with comma
    # remember though, abbreviations als,o have periods. Get rid of some
 
    # if " thle " in speech_text:
    #     print(speech_text.replace(" thle ", " ****thle**** "))
    # then, remaining periods should be mostly the end of sentence info, or those replacing comma's
    period_separated = speech_text.split('.')

    if len(period_separated) < 2:  # to make sure there are at least 2 sentences, or comma separated stuff
        return ''  # not enough sentences to begin with, return just an empty string
    new_text = period_separated[0]

    for piece_index in range(1, len(period_separated)):
        # to check whether the first word of the line following . is starting with uppercase
        first_alpha = re.search(alpha, period_separated[piece_index])
        if len(period_separated[piece_index].strip())>0 and (not first_alpha or first_alpha.group(0).islower()):
            new_text += ' , ' + period_separated[piece_index]
        else:
            new_text += ' . ' + period_separated[piece_index]

    new_text = new_text.replace(escaped_dot, '.').replace(
        '\n', '').replace('  ', ' ')

    # if "today I am introducing a bill to define the affirmative defense of insanity and to establish" in speech_text:
    #     print(f"speech_text: {speech_text}\n\n new: {new_text}")
    seen = set()
    words = new_text.split()
    for (i, word) in enumerate(words):
        w = word.lower().strip('?$"[]()-:,.')
        if len(w)>0 and w not in embeddings_index:
            if w in seen:
                continue
            seen.add(w)
            if not w in oov_counts:
                oov_counts[w] = []
            oov_counts[w].append(" ".join(words[max(0, i-5): min(i+5, len(words)-1)]))
    return new_text

# check the unsuable and irrelevant texts to eliminate them


def shouldExcludeText(speech_text):

    # eliminate few-sentence long texts
    period_separated = speech_text.split(' . ')

    if len(period_separated) < 1:  # remove those speeches with less than 5 sentences long
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
selected_congresses = [97, 100, 106, 114, 112, 109, 112, 114]
out_dir = "../../processed_data2"
# selected_congresses = ['103']
# selected_congresses = [ 114]#range(100, )100,103, 106, 
# selected_congresses = range(98, 115) # 56,

for selected_congress in selected_congresses:
    oov_counts = dict()
    root = "../../congress_data/hein-daily/" if selected_congress >= 97 else "../../congress_data/hein-bound/"
    print(f"processing congress {selected_congress}")

    fmt_congress = "%03d" % selected_congress
    os.makedirs(f'{out_dir}/Senate_{fmt_congress}', exist_ok=True)
    os.makedirs(f'{out_dir}/House_{fmt_congress}', exist_ok=True)

    speech_path = f"{root}/speeches_{fmt_congress}.txt"
    # collect the speeches of this congress here, associated with it's unique speech id
    # includes house, senate speeches, all.
    speech_dict = {}
    thrown_away_ids = set()

    print("Reading speeches...")
    with open(speech_path, 'r', errors='replace') as speech_fo:
        text = speech_fo.read()
    print("do bulk replacements...")
    lines = doBulkReplacements(text).splitlines()
    c = 0
    for line in lines[1:]:# skip the header
        if c % int(len(lines)/20) == 0:
            print(f"{int(c/len(lines) * 100)}%")
        split_line = line.split('|', 1)
        if len(split_line) < 2:
            print("unexpected line @%d: '%s'" %(c+2, line))
            continue
        speech_id = split_line[0]
        original_text = split_line[1]

        # here, preprocess (or rather fix) the text's problems, especially create the correct punctuations
        speech_text = correctText(original_text)

        # add this text to the dictionary only if it is long and relevant enough
        try:
            speech_id = int(speech_id)
        except ValueError:
            continue
        if not shouldExcludeText(speech_text):
            speech_dict[speech_id] = speech_text
            if c in [1, 10, 100, 1000, 2000]:
                print(
                    f"adding {speech_id}: {speech_text}\n from {original_text}")
        else:
            thrown_away_ids.add(speech_id)
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
    df = pd.read_csv(f"{root}{fmt_congress}_SpeakerMap.txt", sep='|', dtype={'party':str, 'firstname':str, 'lastname':str})
    for index in df.index:
        row = df.loc[index].to_dict()
        name = f"{row['firstname']} {row['lastname']}"
        row['name'] = name
        if row['speech_id'] not in speech_dict.keys():
            if row['speech_id'] not in thrown_away_ids:
                print(f"Unexpected speech ID {row['speech_id']}")
        else:
            lower_party = str(row['party']).lower()
            lower_name = name.replace(' ', '_').lower()
            filenamew = f"{lower_party}_{lower_name}_{date_dict[row['speech_id']]}_{row['speech_id']}.json"
            row['speech'] = speech_dict[row['speech_id']]
            row['congress'] = selected_congress
            row['date'] = date_dict[row['speech_id']]

            # start writing up what you read here
            if row['chamber'] == 'H':
                # print('writing house ' + filenamew)
                with open(f'{out_dir}/House_{fmt_congress}/{filenamew}', 'w') as fo_w:
                    fo_w.write(json.dumps(row))
            elif row['chamber'] == 'S':
                # print('writing senate ' + filenamew)
                with open(f'{out_dir}/Senate{ fmt_congress}/{filenamew}', 'w') as fo_w:
                    fo_w.write(json.dumps(row))
            else:
                print(f"Unexpected chamber {row['chamber']}")
    oov_list = sorted(oov_counts.items(),
                      key=lambda kv: len(kv[1]), reverse=True)
    print(f"top missing words:")
    for i in range(50):
        print(
            f"\t'{oov_list[i][0]}' ({len(oov_list[i][1])}), examples: {[x for x in oov_list[i][1][:5]]}")
