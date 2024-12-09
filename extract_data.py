
import csv
import json
import pathlib
from tqdm import tqdm


ORIGINAL_SENTENCES_DIR = './data/sentences'
EXTRACTED_SENTENCES_DIR = './data_extracted/sentences'


def main():
    pathlib.Path(EXTRACTED_SENTENCES_DIR).mkdir(parents=True, exist_ok=True)
    
    print('Started extracting Aesop Fables...')
    extract_aesop_fables()
    print('Finished extracting Aesop Fables.')
    print()

    print('Started extracting Miller Center...')
    extract_miller_center()
    print('Finished extracting Miller Center.')
    print()

    print('Started extracting RateMyProfessor...')
    extract_rate_my_professor()
    print('Finished extracting RateMyProfessor.')
    print()
    
    print('Started extracting ROCStories...')
    extract_roc_stories()
    print('Finished extracting ROCStories.')
    print()


def extract_aesop_fables(original_directory=ORIGINAL_SENTENCES_DIR, extracted_directory=EXTRACTED_SENTENCES_DIR, subdirectory='aesop_fables'):
    pathlib.Path(f'{extracted_directory}/{subdirectory}').mkdir(parents=True, exist_ok=True)
    infilename1 = 'Aesop Fables.json'
    outfilename1 = 'aesop_fables.txt'
    
    with open(f'{original_directory}/{subdirectory}/{infilename1}', 'r') as in_file:
        with open(f'{extracted_directory}/{subdirectory}/{outfilename1}', 'w') as out_file:
            json_file = json.load(in_file)

            is_first_line = True
            for fable in json_file['stories']:
                if is_first_line:
                    is_first_line = False
                else:
                    out_file.write('\n')

                story_str = ' '.join(fable['story'])
                out_file.write(story_str)


def extract_miller_center(original_directory=ORIGINAL_SENTENCES_DIR, extracted_directory=EXTRACTED_SENTENCES_DIR, subdirectory='miller_center'):
    pathlib.Path(f'{extracted_directory}/{subdirectory}').mkdir(parents=True, exist_ok=True)
    infilename1 = 'speeches.txt'
    outfilename1 = 'miller_center.txt'

    with open(f'{original_directory}/{subdirectory}/{infilename1}', 'r') as in_file:
        with open(f'{extracted_directory}/{subdirectory}/{outfilename1}', 'w') as out_file:
            for line in in_file.readlines():
                new_line = line \
                    .replace('<p class="p1">', '') \
                    .replace('<p class="p2">', '') \
                    .replace('<span class="s1">', '') \
                    .replace('<span class="s2">', '') \
                    .replace('</span>', '') \
                    .replace('<br>', ' ') \
                    .replace('&nbsp;', ' ') \
                    .replace('/p&gt;', '') \
                    .replace('&gt;', '') \
                    .replace('&#39;', '\'') \
                    .replace('&#1114;', '') \
                    .replace('&#1029;', '') \
                    .replace('&amp;', '&') \
                    .replace('&quot;', '"') \
                    .replace('&mdash;', '---') \
                    .replace('&deg;', '°') \
                    .replace('&rdquo;', '"') \
                    .replace('&rsquo;', '\'') \
                    .replace('&ldquo', '"') \
                    .replace('&ndash;', '--') \
                    .replace('&frac12;', '1/2') \
                    .replace('<em>', '') \
                    .replace('</em>', '') \
                    .replace('          ', ' ') \
                    .replace('         ', ' ') \
                    .replace('        ', ' ') \
                    .replace('       ', ' ') \
                    .replace('      ', ' ') \
                    .replace('     ', ' ') \
                    .replace('    ', ' ') \
                    .replace('   ', ' ') \
                    .replace('  ', ' ')

                if new_line[0] == ' ':
                    new_line = new_line[1:]
                
                out_file.write(new_line)


def extract_rate_my_professor(original_directory=ORIGINAL_SENTENCES_DIR, extracted_directory=EXTRACTED_SENTENCES_DIR, subdirectory='rate_my_professor'):
    pathlib.Path(f'{extracted_directory}/{subdirectory}').mkdir(parents=True, exist_ok=True)
    infilename1 = 'rmf.csv'
    outfilename1 = 'rate_my_professor.csv'
    
    with open(f'{original_directory}/{subdirectory}/{infilename1}', 'rb') as in_file:
        with open(f'{extracted_directory}/{subdirectory}/{outfilename1}', 'wb') as out_file:
            out_file.writelines(in_file.readlines())


def extract_roc_stories(original_directory=ORIGINAL_SENTENCES_DIR, extracted_directory=EXTRACTED_SENTENCES_DIR, subdirectory='roc_stories'):
    pathlib.Path(f'{extracted_directory}/{subdirectory}').mkdir(parents=True, exist_ok=True)
    infilename1 = 'roc_stories_train.csv'
    outfilename1 = 'roc_stories.txt'
    
    with open(f'{original_directory}/{subdirectory}/{infilename1}', 'r') as in_file:
        with open(f'{extracted_directory}/{subdirectory}/{outfilename1}', 'w') as out_file:
            csv_file = csv.DictReader(in_file)

            is_first_line = True
            for line in csv_file:
                if is_first_line:
                    is_first_line = False
                else:
                    out_file.write('\n')

                just_the_text = line['sentence1'] + ' ' + line['sentence2'] + ' ' + line['sentence3'] + ' ' + line['sentence4'] + ' ' + line['sentence5']
                out_file.write(just_the_text)


if __name__ == '__main__':
    main()
