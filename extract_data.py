
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
