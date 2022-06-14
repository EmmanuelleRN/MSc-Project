# -*- coding: utf-8 -*-
"""
Created on Sat May 28 15:07:12 2022

@author: Emmanuelle R Nunes
"""

import requests
import steamreviews
from bs4 import BeautifulSoup

def get_n_appids(n=100, filter_by='topsellers'):
    app_ids = []
    url = f'https://store.steampowered.com/search/?category1=998&filter={filter_by}&page='
    page = 0

    while page*25 < n:
        page += 1
        response = requests.get(url=url+str(page), headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        for row in soup.find_all(class_='search_result_row'):
            app_ids.append(row['data-ds-appid'])

    return app_ids[:n]

app_ids = get_n_appids(750)


request_params = dict()
request_params['day_range'] = '30'
request_params['language'] = 'english'

steamreviews.download_reviews_for_app_id_batch(app_ids, 
                                               chosen_request_params=request_params)