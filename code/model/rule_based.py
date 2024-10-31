from bs4 import BeautifulSoup, Comment, Doctype
import pickle
from helper import *
from tqdm import tqdm
from rule_based import ValidAnswerExtractor
from lxml import html
import statistics
import pandas as pd

import sys
sys.path.append(
    '/Users/andrewfong/Downloads/eBay_ML_Challenge_2024/code/analysis')
from combo_analysis import calculate_scores

def remove_enclosed_substrings(snippet, pairs):
    stack = []
    to_remove = []
    opening_chars = {opening for opening, _ in pairs}
    closing_chars = {closing for _, closing in pairs}
    pair_map = {opening: closing for opening, closing in pairs}
    closing_to_opening = {closing: opening for opening, closing in pairs}

    snippet_length = len(snippet)
    i = 0
    while i < snippet_length:
        char = snippet[i]
        if char in opening_chars:
            # Push the opening char and its index onto the stack
            stack.append((char, i))
        elif char in closing_chars:
            if stack and stack[-1][0] == closing_to_opening[char]:
                # We have a matching pair
                opening_char, opening_index = stack.pop()
                closing_index = i
                to_remove.append((opening_index, closing_index))
            else:
                # Unmatched closing character
                pass  # Do nothing
        else:
            pass  # Do nothing
        i += 1

    # Now, we need to remove the substrings in to_remove
    # We need to process the ranges in reverse order so that indices remain valid
    # First, merge overlapping or nested ranges
    to_remove = sorted(to_remove, key=lambda x: x[0])
    merged = []
    for start, end in to_remove:
        if merged and start <= merged[-1][1]:
            # Overlapping or nested ranges
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Build the result string by excluding the ranges to remove
    result = []
    prev_index = 0
    for start, end in merged:
        result.append(snippet[prev_index:start])
        prev_index = end + 1  # Skip the closing character
    result.append(snippet[prev_index:])

    return ''.join(result)

# include headers


def extract_text_snippets(html_content: str, num_char: int, require_num: int = 5):
    snippets = []

    # Parse the HTML content with BeautifulSoup using 'lxml' parser
    soup = BeautifulSoup(html_content, 'lxml')

    # Define tags to exclude (like scripts, styles, links, etc.)
    blacklist = {'script', 'style', 'head',
                 'meta', 'title', 'link', 'noscript', 'a'}

    # Remove comments to prevent extraction of commented-out code
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove DOCTYPE declarations to prevent extraction of DOCTYPE text
    for doctype in soup.find_all(string=lambda text: isinstance(text, Doctype)):
        doctype.extract()

    # Handle tables individually
    tables = soup.find_all('table')
    tables_to_remove = []
    for table in tables:
        # Find the header row
        header_row = None

        # Check for <thead>
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')

        # If no header row found in <thead>, look for the first <tr> in the table
        if not header_row:
            header_row = table.find('tr')

        # Proceed only if a header row is found
        if header_row:
            # Extract header text
            header_cells = header_row.find_all(['th', 'td'])
            header_texts = [text_normalization(
                cell.get_text(strip=True)) for cell in header_cells]
            header_set = set(header_texts)

            # Define required columns as a list of tuples
            # Each tuple contains alternative column names
            required_columns = [
                ('year', 'years')         # Accepts 'year' or 'years'
            ]

            # Check if the header contains at least one column from each tuple
            if all(any(col in header_set for col in alternatives) for alternatives in required_columns):
                # Process table rows
                for tr in table.find_all('tr'):
                    # Skip the header row
                    if tr == header_row:
                        continue

                    # Extract the text content of the entire row
                    row_cells = tr.find_all(['th', 'td'])
                    row_texts = [cell.get_text(
                        separator=' ', strip=True) for cell in row_cells]
                    # Remove any extra whitespace or line breaks within cell texts
                    row_texts = [' '.join(text.split()) for text in row_texts]
                    row_text = ' '.join(row_texts).strip()
                    if 'whi solutions' in row_text.lower():
                        print('whi solutions found in table!!!')
                        # Early exit if forbidden text is found
                        return (True, [])

                    if len(row_text) >= require_num:
                        # Extract the first num_char characters
                        snippet = row_text[:num_char]
                        # Remove enclosed substrings
                        snippet = remove_enclosed_substrings(
                            snippet, [('(', ')')])
                        snippets.append(snippet)
                # Mark this table for removal after processing
                tables_to_remove.append(table)
            else:
                # Do not process this table; treat it as regular text
                continue
        else:
            # No header row found; treat table as regular text
            continue

    # Remove processed tables from the soup to avoid duplicate processing
    for table in tables_to_remove:
        table.extract()

    # Find all text nodes not inside processed tables
    texts = soup.find_all(string=True)
    for text_node in texts:
        # Skip text containing 'whi solutions' (case-insensitive)
        if 'whi solutions' in text_node.lower():
            return (True, [])  # Early exit if forbidden text is found

        # Check if any ancestor tag is in the blacklist
        if any(parent.name in blacklist for parent in text_node.parents):
            continue  # Skip this text node

        # Skip text nodes inside processed tables
        if any(parent in tables_to_remove for parent in text_node.parents):
            continue  # Skip this text node

        # Clean and process the text
        content = text_node.strip()
        if not content:
            continue  # Skip empty strings

        # Instead of splitting into lines, treat the content as a whole
        if len(content) >= require_num:
            # Extract the first num_char characters
            snippet = content[:num_char]
            # Remove enclosed substrings
            snippet = remove_enclosed_substrings(snippet, [('(', ')')])
            snippets.append(snippet)

    return (False, snippets)


def replace_nicknames(normalized_str):
    nickename_to_fullname_normalized = {
        'f250sd': 'f250 super duty',
        'f350sd': 'f350 super duty',
        'f450sd': 'f450 super duty',
        'f550sd': 'f550 super duty',
        'chevy': 'chevrolet',
        'vw': 'volkswagen'
    }
    for nickname, fullname in nickename_to_fullname_normalized.items():
        normalized_str = normalized_str.replace(nickname, fullname)
    return normalized_str


def get_comparator_key(item):
    year, make, model = item.split(',')
    return (year, make, model)


def is_classic(make_model):
    classics = {'Chevrolet,Silverado 1500 Classic',
                'Chevrolet,Silverado 1500 HD Classic',
                'Chevrolet,Silverado 2500 HD Classic',
                'Chevrolet,Silverado 3500 Classic',
                'GMC,Sierra 1500 Classic',
                'GMC,Sierra 1500 HD Classic',
                'GMC,Sierra 2500 HD Classic',
                'GMC,Sierra 3500 Classic'}
    return make_model in classics


is_train_set = False
banning_models = True
banning_brands = True
banning_years = True
banning_types = True
use_bigger_answer_bank = False

model_prob_req = 0.55
model_size_req = 2

brand_prob_req = 0.5
brand_size_req = 5

years_prob_req = 0.5
years_size_req = 2

type_prob_req = 0.45
type_size_req = 5

model_given_make_req = 0.8

if is_train_set:
    debug_file = 'rule_based_debug.txt'
    output_file = 'rule_based_output.txt'
    debug_no_correct_file = 'rule_based_debug_no_correct.txt'
    debug_mistake_file = 'rule_based_debug_mistake.txt'
    conditioned_file = 'rule_based_conditioned.txt'

    debug_no_correct_file = open(debug_no_correct_file, 'w')
    debug_mistake_file = open(debug_mistake_file, 'w')
    conditioned_file = open(conditioned_file, 'w')
else:
    debug_file = 'rule_based_debug_quiz.txt'
    output_file = 'rule_based_output_quiz.txt'

debug_file = open(debug_file, 'w')
output_file = open(output_file, 'w')

with open(f'/Users/andrewfong/Downloads/eBay_ML_Challenge_2024/{'serialized' if is_train_set else 'serialized_quiz'}.pkl', 'rb') as file:
    data = pickle.load(file)

with open('gen/gen_results/model_make_probs.pkl', 'rb') as file:
    model_make_probs = pickle.load(file)

# with open('./years_probs.pkl', 'rb') as file:
#     years_prob = pickle.load(file)

for makes in model_make_probs.values():
    del makes['total']

tags = data['tags']
items = data['items']
html_descriptions = data['html_descriptions']
solutions = data['solutions']

banned_models_make_present = set()
banned_models_make_not_present = set()

if banning_models:
    with open(('gen/gen_results/model_correctness_make_present.pkl' if not use_bigger_answer_bank else 'gen/gen_results/model_correctness_make_present_complete.pkl'), 'rb') as file:
        model_correctness_make_present = pickle.load(file)

    with open(('gen/gen_results/model_correctness_make_not_present.pkl' if not use_bigger_answer_bank else 'gen/gen_results/model_correctness_make_not_present_complete.pkl'), 'rb') as file:
        model_correctness_make_not_present = pickle.load(file)

    for model in model_correctness_make_present:
        if model_correctness_make_present[model][0] < model_prob_req and model_correctness_make_present[model][1] >= model_size_req:
            banned_models_make_present.add(model)

    for model in model_correctness_make_not_present:
        if model_correctness_make_not_present[model][0] < model_prob_req and model_correctness_make_not_present[model][1] >= model_size_req:
            banned_models_make_not_present.add(model)

banned_brands = set()

if banning_brands:
    with open('gen/gen_results/brand_correctness.pkl', 'rb') as file:
        brand_correctness = pickle.load(file)
    for brand in brand_correctness:
        if brand_correctness[brand][0] < brand_prob_req and brand_correctness[brand][1] >= brand_size_req:
            banned_brands.add(brand)

banned_types = set()

if banning_types:
    with open('gen/gen_results/type_correctness.pkl', 'rb') as file:
        type_correctness = pickle.load(file)
    for part_type in type_correctness:
        if type_correctness[part_type][0] < type_prob_req and type_correctness[part_type][1] >= type_size_req:
            banned_types.add(part_type)

banned_years = {}

if banning_years:
    with open('gen/gen_results/years_correctness_given_make_model.pkl', 'rb') as file:
        years_correctness_make_model = pickle.load(file)
    for make_model in years_correctness_make_model:
        banned_years[make_model] = set()
        for year in years_correctness_make_model[make_model]:
            if years_correctness_make_model[make_model][year][0] < years_prob_req and years_correctness_make_model[make_model][year][1] >= years_size_req:
                banned_years[make_model].add(year)

if use_bigger_answer_bank:
    with open('complete_answer_bank.pkl', 'rb') as file:
        answer_bank = pickle.load(file)
else:
    answer_bank = data['answer_bank']
    answer_bank = set(answer_bank)

best_make_given_model = {}

for model in model_make_probs:
    for make in model_make_probs[model]:
        best_make, prob = max(
            model_make_probs[model].items(), key=lambda x: x[1])
        if prob >= model_given_make_req:
            best_make_given_model[model] = best_make

normalized_no_space_to_answer = {text_normalization(
    answer).replace(' ', ''): answer for answer in answer_bank}

# low probability of a model containing certain year and that year also being present in the sentence

extractor = ValidAnswerExtractor(
    answer_bank, best_make_given_model, banned_models_make_not_present, banned_models_make_present)

predicted_answer = {i: set() for i in (
    range(5000, 30000) if not is_train_set else range(5000))}

total_answer = 0
correct = 0

for i in tqdm(range(5000, 30000) if not is_train_set else range(5000)):
    debug_file.write(f'{i}\n\n')
    if is_train_set:
        debug_mistake_file.write(f'{i}\n')
        debug_no_correct_file.write(f'{i}\n\n')

    brand = get_attribute_value(tags[i], ['brand', 'brands'])
    brand = text_normalization(brand)

    if brand in banned_brands:
        debug_file.write(f'{brand} banned\n')
        if is_train_set:debug_no_correct_file.write(f'{brand} banned\n')
        continue

    part_type = get_attribute_value(tags[i], ['type', 'part type'])
    part_type = text_normalization(part_type)

    if part_type in banned_types:
        debug_file.write(f'{part_type} banned\n')
        if is_train_set: debug_no_correct_file.write(f'{part_type} banned\n')
        continue

    haswhi, snippets = extract_text_snippets(html_descriptions[i], 80)

    if not snippets:
        if not haswhi:
            continue
        years, makes, models = extract_vehicle_info(html_descriptions[i])

        if not years or not makes or not models:
            continue

        years = sorted(list(years))
        valid_make_models = set()

        for make in makes:
            for model in models:
                make_model = make + model
                make_model = normalized_no_space_to_answer[
                    make_model] if make_model in normalized_no_space_to_answer else ""
                if make_model in answer_bank:
                    valid_make_models.add(make_model)

        whi_solutions = set()
        if len(valid_make_models) <= 3:
            for make_model in valid_make_models:
                for year in years:
                    if is_classic(make_model):
                        whi_solutions.add(f'{True}~*~{2007},{make_model}')
                        break
                    whi_solutions.add(f'{True}~*~{year},{make_model}')
        elif len(valid_make_models) * len(years) <= 25:
            if len(years) <= 2:
                continue
            for year in years[1:-1]:
                for make_model in valid_make_models:
                    if is_classic(make_model):
                        whi_solutions.add(f'{True}~*~{2007},{make_model}')
                        continue
                    whi_solutions.add(f'{True}~*~{year},{make_model}')
        # else:
        #     if len(years) <= 2: continue
        #     mid_year = years[len(years) // 2]
        #     for make_model in valid_make_models:
        #         whi_solutions.add(f'{True}~*~{mid_year},{make_model}')
        #         if f'{mid_year},{make_model}' in solutions[i]:
        #             correct += 1
        #         total_answer += 1
        # else:
        #     num_possible_answers = len(valid_make_models) * len(years)
        #     num_unilateral_truncation = round(0.003 * num_possible_answers ** 2)
        #     years = sorted(set(years))
        #     if num_unilateral_truncation * 2 > len(years): years = []
        #     else: years = years[num_unilateral_truncation:-num_unilateral_truncation]

        #     for make_model in valid_make_models:
        #         for year in years:
        #             predicted_answer[i].add(f'{True}~*~{year},{make_model}')
        #             if f'{year},{make_model}' in solutions[i]:
        #                 correct += 1
        #             total_answer += 1

        # for valid_make_model in valid_make_models:
        #     loca_years_prob = {}
        #     for year in years:
        #         if year in years_prob[valid_make_model]: loca_years_prob[year] = years_prob[valid_make_model][year]
        #         else: loca_years_prob[year] = 1 / len(years_prob[valid_make_model])
        #     valid_years = filter_keys_by_std(loca_years_prob, 2)
        #     for year in valid_years:
        #         pass
        #         #predicted_answer[i].add(f'{year},{valid_make_model}')


        for answer_and_condition in whi_solutions:
            _, answer = answer_and_condition.split('~*~')
            year, make, model = answer.split(',')
            make_model = make + ',' + model
            if make_model not in banned_models_make_present and (make_model not in banned_years or year not in banned_years[make_model]):
                predicted_answer[i].add(answer_and_condition)
                debug_file.write(f'{year},{make_model} added\n')
                if is_train_set and f'{year},{make_model}' not in solutions[i]:
                    debug_no_correct_file.write(f'{year},{make_model} added\n')
                    debug_mistake_file.write(f'{year},{make_model} added\n')

        continue

    snippets.append(items[i][1])

    for sni in snippets:
        if '..' in sni:
            debug_file.write(f'{sni} removed cuz of ..\n')
            if is_train_set:
                debug_no_correct_file.write(f'{sni} removed cuz of ..\n')
            continue
        sni = sni.strip()
        sni = sni.replace('\xa0', '')
        sni = extract_or_return_first_n_words(sni, 100)

        years = get_years(sni)
        extractee = ''
        make_present = None

        incorrect = False

        if len(years) != 0:
            make_present, extractee = extractor.get_answer(replace_nicknames(
                # normalization here is redundant but acceptable
                text_normalization(sni)))
            if extractee and extractee != 'nuked' and 'ambiguous makes for ' not in extractee:
                for year in years:
                    if extractee in banned_years and year in banned_years[extractee]:
                        extractee = 'banned year'
                        break
                    predicted_answer[i].add(
                        f'{make_present}~*~{year},{extractee}')
                    if is_train_set and f'{year},{extractee}' not in solutions[i]:
                        incorrect = True
        else:
            extractee = 'no years'

        if make_present == True:
            make_debug_text = 'true'
        elif make_present == False:
            make_debug_text = 'false'
        else:
            make_debug_text = 'n/a'

        debug_file.write(
            f'{sni}||||{extractee}({make_debug_text})||||{years}\n')

        if is_train_set:
            if len(extractee.split(',')) != 2:
                debug_no_correct_file.write(
                    f'{sni}||||{extractee}({make_debug_text})||||{years}\n')
            elif incorrect:
                debug_mistake_file.write(
                    f'{sni}||||{extractee}({make_debug_text})||||{years}\n')
                debug_no_correct_file.write(
                    f'{sni}||||{extractee}({make_debug_text})||||{years}\n')

if is_train_set:
    for i in (range(5000, 30000) if not is_train_set else range(5000)):
        conditioned_file.write('|'.join(predicted_answer[i]) + '\n')

predicted_answer = {i: {answer.split(
    '~*~')[1] for answer in predicted_answer[i]} for i in predicted_answer}

for i in (range(5000, 30000) if not is_train_set else range(5000)):
    output_file.write('|'.join(predicted_answer[i]) + '\n')

debug_file.close()
output_file.close()

if is_train_set:
    conditioned_file.close()
    debug_no_correct_file.close()
    debug_mistake_file.close()

if is_train_set:
    calculate_scores(['rule_based_output.txt'], [0], '|')
    exit()

submission_dict = {'RECORD_ID': [], 'FTMNT_YEAR': [],
                   'FTMNT_MAKE': [], 'FTMNT_MODEL': []}

for i in range(5000, 30000):
    for answer in predicted_answer[i]:
        year, make, model = answer.split(',')
        submission_dict['RECORD_ID'].append(i)
        submission_dict['FTMNT_YEAR'].append(year)
        submission_dict['FTMNT_MAKE'].append(make)
        submission_dict['FTMNT_MODEL'].append(model)

submission_df = pd.DataFrame(submission_dict)
submission_df.to_csv('submission.csv.gz', index=False, compression='gzip')
