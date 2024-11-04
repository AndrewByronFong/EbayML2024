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

from rule_based_discriminator import Discriminator
from rule_based_printer import Printer
from whi_extractor import Whi_Extractor
from data_loader import Data_Loader
from rule_based_generator import Generator

# note that all set level will run but level 2 will not yield any output even though the results are computed

set_level = 2
banning_models = banning_brands = banning_years = banning_types = True
training_generator = True

model_prob_req = 0.55
model_size_req = 2

brand_prob_req = 0.5
brand_size_req = 5

years_prob_req = 0.5
years_size_req = 2

type_prob_req = 0.45
type_size_req = 5

model_given_make_req = 0.8
generator_threshold = 0.5

mpn = ['mpn', 'manufacturer part number', 'manufacturer part ']
oem = ['oeoem part number']
ipn = ['interchange part number']
other = ['other part number']

pn = [mpn, oem, ipn, other]

best_make_given_model = get_best_make_given_model(model_given_make_req)

data_loader = Data_Loader(set_level)
answer_bank = data_loader.get_answer_bank()

extractor = ValidAnswerExtractor(
    answer_bank, best_make_given_model)

whi_extractor = Whi_Extractor(answer_bank)

discriminator = Discriminator(
    (banning_models, model_prob_req, model_size_req),
    (banning_brands, brand_prob_req, brand_size_req),
    (banning_years, years_prob_req, years_size_req),
    (banning_types, type_prob_req, type_size_req)
)

printer = Printer(set_level)

generator = Generator(training_generator, generator_threshold, pn)

normalized_to_answerbank = {text_normalization(answer): answer for answer in answer_bank}

is_train_set = set_level == 0

if set_level == 0: itr = range(5000)
elif set_level == 1: itr = range(5000, 30000)
else: itr = range(4747128) # maybe change to 5000000 later

#predicted_answer = {i: set() for i in itr}

for i in tqdm(itr):
    predicted_answer = set()
    if not training_generator:
        generator_output = generator.get(data_loader.get_tag(i))
        if generator_output:
            for extractee in generator_output:
                #predicted_answer[i].add(f'N/A~*~{extractee}') # note that the make_present is really N/A
                pass
            continue
    
    # vin = get_attribute_value(data_loader.get_tag(i), ['vin', 'vin ', 'vehicle vin'])
    # if vin: 
    #     answer = decode_vin(vin)
    #     year, make, model = answer.split(',')
    #     make_model = make + ',' + model
    #     normalized_make_model = text_normalization(make_model)
    #     if normalized_make_model in normalized_to_answerbank: predicted_answer[i].add(f'N/A~*~{year},{normalized_to_answerbank[normalized_make_model]}')
    
    # mpn = get_part_number(data_loader.get_tag(i), ['mpn', 'manufacturer part number', 'manufacturer part '], ['does', 'not', 'apply'])
    # if all([mpn != mpn_control for mpn_control in ['8w7e12a366aa', '2731al60c', 'uf622', 'fd117t', '25430plr003']]):
    #     continue

    printer.print_debug(f'{i}\n')
    if is_train_set: printer.print_debug_mistake(f'{i}\n')
    
    brand = get_attribute_value(data_loader.get_tag(i), ['brand', 'brands'])
    brand = text_normalization(brand)

    if discriminator.is_brand_banned(brand):
        printer.print_debug(f'{brand} banned')
        continue

    part_type = get_attribute_value(data_loader.get_tag(i), ['type', 'part type'])
    part_type = text_normalization(part_type)

    if discriminator.is_type_banned(part_type):
        printer.print_debug(f'{part_type} banned')
        continue

    haswhi, haslkq, snippets = extract_text_snippets(data_loader.get_desc(i), 80)

    if haswhi:
        if (is_train_set and (not banning_brands or not banning_types or not banning_models or not banning_years)): # preventing whi from damaging the banned list distribution
            continue
        
        whi_solutions = whi_extractor.extract(data_loader.get_desc(i))

        for answer in whi_solutions:
            year, make, model = answer.split(',')
            make_model = make + ',' + model
            if not discriminator.is_model_banned(make_model, True) and not discriminator.is_year_banned(make_model, year):
                #predicted_answer[i].add(f'True~*~{answer}')
                predicted_answer.add(answer)
                printer.print_debug(f'{year},{make_model}')
                if is_train_set and answer not in data_loader.get_solution(i): printer.print_debug_mistake(f'{year},{make_model}')
            else: printer.print_debug(f'{year},{make_model} banned')
        
        generator.train(data_loader.get_tag(i), '|'.join(predicted_answer))
        continue

    if not haslkq: snippets.append(data_loader.get_item(i)[1])
    else: snippets = [data_loader.get_item(i)[1]]

    for sni in snippets:
        if '..' in sni:
            printer.print_debug(f'{sni} removed cuz of ..')
            continue
        sni = sni.strip()
        sni = sni.replace('\xa0', '')
        sni = extract_or_return_first_n_words(sni, 100)

        years = get_years(sni)
        extractee = ''
        make_present = None
        debug_text = ""
        incorrect = False

        if len(years) != 0:
            make_present, num_options, extractee = extractor.get_answer(replace_nicknames(
                # normalization here is redundant but acceptable
                text_normalization(sni)))
            if extractee and 'ambiguous makes for ' not in extractee:
                if discriminator.is_model_banned(extractee, make_present): extractee = 'banned model'
                else:   
                    for year in years:
                        if discriminator.is_year_banned(extractee, year): debug_text += f'{year} banned '
                        else: 
                            #predicted_answer[i].add(f'{make_present}~*~{year},{extractee}')
                            predicted_answer.add(f'{year},{extractee}')
                            if is_train_set and f'{year},{extractee}' not in data_loader.get_solution(i): incorrect = True
        else:
            extractee = 'no years'

        if make_present == True: make_debug_text = 'true'
        elif make_present == False: make_debug_text = 'false'
        else: make_debug_text = num_options = 'n/a'

        printer.print_debug(f'{sni}||||{extractee}({make_debug_text})({num_options})||||{years}{'||||' + debug_text if debug_text else ""}')
        if incorrect and is_train_set: printer.print_debug_mistake(f'{sni}||||{extractee}({make_debug_text})({num_options})||||{years}{'||||' + debug_text if debug_text else ""}')

    generator.train(data_loader.get_tag(i), '|'.join(predicted_answer))
    if i % 25000 == 0: generator.save()
generator.save()
exit()

for i in itr:
    printer.print_conditioned('|'.join(predicted_answer[i]))

predicted_answer = {i: {answer.split(
    '~*~')[1] for answer in predicted_answer[i]} for i in predicted_answer}

for i in itr:
    printer.print_output('|'.join(predicted_answer[i]))
    if training_generator: generator.train(data_loader.get_tag(i), '|'.join(predicted_answer[i]))

if training_generator: generator.save()
printer.flush()

if is_train_set:
    calculate_scores(['model_output/rule_based_output.txt'], [0], '|')
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
submission_df.to_csv('submissions/submission.csv.gz', index=False, compression='gzip')
