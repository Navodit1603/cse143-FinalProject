
import csv
import json
import pathlib
from tqdm import tqdm
from unidecode import unidecode
import random


ORIGINAL_SENTENCES_DIR = './data/sentences'
EXTRACTED_SENTENCES_DIR = './data_extracted/sentences'


def main():
    pathlib.Path(EXTRACTED_SENTENCES_DIR).mkdir(parents=True, exist_ok=True)
    
    # print('Started extracting Aesop Fables...')
    # extract_aesop_fables()
    # print('Finished extracting Aesop Fables.')
    # print()

    # print('Started extracting Miller Center...')
    # extract_miller_center()
    # print('Finished extracting Miller Center.')
    # print()

    # print('Started extracting RateMyProfessor...')
    # extract_rate_my_professor()
    # print('Finished extracting RateMyProfessor.')
    # print()
    
    # print('Started extracting ROCStories...')
    # extract_roc_stories()
    # print('Finished extracting ROCStories.')
    # print()

    # print('Started extracting ROCStories...')
    # extract_roc_stories()
    # print('Finished extracting ROCStories.')
    # print()

    print('Started extracting Wikipedia Stories...')
    extract_wikipedia_sentences()
    print('Finished extracting Wikipedia Stories.')
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
    infilename1 = 'speeches.json'
    outfilename1 = 'miller_center.csv'

    csv_data = []

    with open(f'{original_directory}/{subdirectory}/{infilename1}', 'r') as in_file:
            json_file = json.load(in_file)

            for speech in json_file:
                transcript: list[str] = list(speech['transcript'])

                for i in range(0, len(transcript)):
                    if ord(transcript[i]) < 32 or ord(transcript[i]) == 127:
                        transcript[i] = ' '

                transcript_str = unidecode(''.join(transcript)) \
                    .replace('\n', ' ') \
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

                if transcript_str[0] == ' ':
                    transcript_str = transcript_str[1:]
                
                president_str = unidecode(speech['president'])
                date_str = unidecode(speech['date'])
                title_str = unidecode(speech['title'])

                csv_data.append({
                    'president': president_str,
                    'date': date_str,
                    'title': title_str,
                    'transcript': transcript_str
                })
    
    with open(f'{extracted_directory}/{subdirectory}/{outfilename1}', 'w') as out_file:
        csv_writer = csv.DictWriter(out_file, fieldnames=['president', 'date', 'title', 'transcript'])
        csv_writer.writeheader()
        csv_writer.writerows(csv_data)


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

def extract_wikipedia_sentences(original_directory=ORIGINAL_SENTENCES_DIR, extracted_directory=EXTRACTED_SENTENCES_DIR, subdirectory='wikipedia'):
    pathlib.Path(f'{extracted_directory}/{subdirectory}').mkdir(parents=True, exist_ok=True)
    infilename1 = 'wikisent2.txt'
    outfilename1 = 'wikipedia_sentences.txt'
    
    with open(f'{original_directory}/{subdirectory}/{infilename1}', 'rb') as in_file:
        with open(f'{extracted_directory}/{subdirectory}/{outfilename1}', 'wb') as out_file:
            lines = []

            line_read = in_file.readline()
            while line_read:
                lines.append(line_read)
                line_read = in_file.readline()
            
            random.seed(12345)
            random.shuffle(lines)

            out_file.writelines(lines)


if __name__ == '__main__':
    main()
