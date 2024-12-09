
import pathlib
import requests
import json
from tqdm import tqdm

SENTENCES_DIR = './data/sentences'

def main():
    pathlib.Path(SENTENCES_DIR).mkdir(parents=True, exist_ok=True)

    print('Started miller center...')
    download_miller_center()
    print('Finished miller center.')

def download_miller_center(directory=SENTENCES_DIR, subdirectory='miller_center', name='speeches.txt'):
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
        # print(f'{len(items)} speeches')
    
    with open(f'{directory}/{subdirectory}/{name}', "w") as out_file:
        for item in items:
            transcript: list[str] = list(item['transcript'])
            for i in range(0, len(transcript)):
                if ord(transcript[i]) < 32 or ord(transcript[i]) == 127:
                    transcript[i] = ' '

            transcript_str = ''.join(transcript) \
                .replace('<p class="p1">', '') \
                .replace('<span class="s1">', '') \
                .replace('</span>', '') \
                .replace('<br>', ' ') \
                .replace('&nbsp;', ' ') \
                .replace('/p&gt;', '') \
                .replace('&gt;', '') \
                .replace('&#39;', '\'') \
                .replace('&amp;', '&') \
                .replace('&quot;', '') \
                .replace('&mdash;', '') \
                .replace('&deg;', '') \
                .replace('&rdquo;', '') \
                .replace('&rsquo;', '') \
                .replace('&ldquo', '') \
                .replace('&ndash;', '') \
                .replace('&frac12;', '') \
                .replace('&c.;', '') \
                .replace('&c.', '.') \
                .replace('<em>', '') \
                .replace('</em>', '')
            
            out_file.write(transcript_str)
            out_file.write('\n')

def download_rate_my_prof(directory=SENTENCES_DIR, name='rmf.csv'):
    r: requests.Response = requests.get('https://data.mendeley.com/public-files/datasets/fvtfjyvw7d/files/256a4429-4fc3-4872-9a7c-26b44a820a8c/file_downloaded')
    print(len(r.content))

if __name__ == '__main__':
    main()
