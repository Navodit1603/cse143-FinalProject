
import pathlib
import requests
from tqdm import tqdm
import kagglehub


SENTENCES_DIR = './data/sentences'


def main():
    pathlib.Path(SENTENCES_DIR).mkdir(parents=True, exist_ok=True)

    print('Started Aesop Fables...')
    download_aesop_fables()
    print('Finished Aesop Fables.')
    print()

    print('Started Miller Center...')
    download_miller_center()
    print('Finished Miller Center.')
    print()

    print('Started RateMyProfessor...')
    download_rate_my_prof()
    print('Finished RateMyProfessor.')
    print()

    # roc_download_instructions()
    print('Started ROCStories...')
    download_roc_stories()
    print('Finished ROCStories.')
    print()

    print('Finished downloading all data.')


def roc_download_instructions(directory=SENTENCES_DIR, subdirectory='roc'):
    pathlib.Path(f'{directory}/{subdirectory}').mkdir(parents=True, exist_ok=True)

    print('For the ROCStories dataset, please download the files manually.')
    print('Here are the Google Drive links to each file:')
    print(' - https://drive.google.com/file/d/1LoLfsRHwwEFbnyW8qJ9aZS5IYiRRZWDS/view?usp=sharing')
    print(' - https://drive.google.com/file/d/1iIMKXGwHlMtXwE-35LVkaiNo4rQ3RriK/view?usp=sharing')
    print('Please place each file in \'./data/roc_stories\', and preserve the original file name.')


def download_roc_stories(directory=SENTENCES_DIR, subdirectory='roc_stories'):
    outfilename1 = 'roc_stories_train.csv'
    outfilename2 = 'roc_stories_test.csv'
    pathlib.Path(f'{directory}/{subdirectory}').mkdir(parents=True, exist_ok=True)
    
    response = requests.get('https://drive.usercontent.google.com/uc?id=1LoLfsRHwwEFbnyW8qJ9aZS5IYiRRZWDS&export=download', stream=True)
    with open(f'{directory}/{subdirectory}/{outfilename1}', 'wb') as out_file:
        for data in tqdm(response.iter_content()):
            out_file.write(data)
    
    response = requests.get('https://drive.usercontent.google.com/uc?id=1iIMKXGwHlMtXwE-35LVkaiNo4rQ3RriK&export=download', stream=True)
    with open(f'{directory}/{subdirectory}/{outfilename2}', 'wb') as out_file:
        for data in tqdm(response.iter_content()):
            out_file.write(data)


def download_aesop_fables(directory=SENTENCES_DIR, subdirectory='aesop_fables'):
    outfilename = 'Aesop Fables.json'
    pathlib.Path(f'{directory}/{subdirectory}').mkdir(parents=True, exist_ok=True)

    kaggle_path = kagglehub.dataset_download('muhammadardiputra/aesop-fables-dataset')
    
    with open(f'{kaggle_path}/{outfilename}', 'rb') as in_file:
        with open(f'{directory}/{subdirectory}/{outfilename}', 'wb') as out_file:
            out_file.writelines(in_file.readlines())


def download_children_stories(directory=SENTENCES_DIR, subdirectory='children_stories'):
    outfilename1 = 'children_books.csv'
    outfilename2 = 'children_stories.Csv'
    pathlib.Path(f'{directory}/{subdirectory}').mkdir(parents=True, exist_ok=True)

    kaggle_path = kagglehub.dataset_download('thomaskonstantin/highly-rated-children-books-and-stories')
    
    with open(f'{kaggle_path}/{outfilename1}', 'rb') as in_file:
        with open(f'{directory}/{subdirectory}/{outfilename1}', 'wb') as out_file:
            out_file.writelines(in_file.readlines())
    
    with open(f'{kaggle_path}/{outfilename2}', 'rb') as in_file:
        with open(f'{directory}/{subdirectory}/{outfilename2}', 'wb') as out_file:
            out_file.writelines(in_file.readlines())


def download_rate_my_prof(directory=SENTENCES_DIR, subdirectory='rate_my_professor'):
    outfilename='rmf.csv'
    pathlib.Path(f'{directory}/{subdirectory}').mkdir(parents=True, exist_ok=True)
    
    response = requests.get('https://data.mendeley.com/public-files/datasets/fvtfjyvw7d/files/256a4429-4fc3-4872-9a7c-26b44a820a8c/file_downloaded', stream=True)
    with open(f'{directory}/{subdirectory}/{outfilename}', 'wb') as out_file:
        for data in tqdm(response.iter_content()):
            out_file.write(data)


def download_miller_center(directory=SENTENCES_DIR, subdirectory='miller_center'):
    outfilename='speeches.txt'
    pathlib.Path(f'{directory}/{subdirectory}').mkdir(parents=True, exist_ok=True)

    the_url = 'https://api.millercenter.org/speeches'

    request = requests.post(url=the_url)
    data = request.json()
    items = data['Items']

    while 'LastEvaluatedKey' in data:
        parameters = {"LastEvaluatedKey": data['LastEvaluatedKey']['doc_name']}
        request = requests.post(url=the_url, params=parameters)
        data = request.json()
        items += data['Items']
        print(f'{len(items)} speeches')
    
    with open(f'{directory}/{subdirectory}/{outfilename}', "w") as out_file:
        is_first_line = True
        for item in items:
            if is_first_line:
                is_first_line = False
            else:
                out_file.write('\n')

            transcript: list[str] = list(item['transcript'])
            for i in range(0, len(transcript)):
                if ord(transcript[i]) < 32 or ord(transcript[i]) == 127:
                    transcript[i] = ' '

            transcript_str = ''.join(transcript)
            
            out_file.write(transcript_str)


if __name__ == '__main__':
    main()
