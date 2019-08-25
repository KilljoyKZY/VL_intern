'''
This code just iterate through the campaign data of our sellers and generate a summary of our sellers.
'''

import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = 100

import pandas as pd
import sys
import time
# sys.path.insert(0,'../')
# from importlib import reload
# import PWLib
# reload(PWLib)


import urllib
# sys.path.insert(0,r'Desktop\vl_code\DataScience')
sys.path.insert(0,r'C:\Users\Zhaoyu Kou\Anaconda3\Lib\DataScience')

import numpy as np
from multiprocessing import Pool, current_process
from importlib import reload
import DataScience.PWLib as PWLib
reload(PWLib)
from PWLib.VTracker import VTracker

vt_key='input the v tracker key here'

reload(PWLib)
from PWLib import AzureBlob
from PWLib import VL_Blobs
from PWLib.VL_Blobs import ASIN
from PWLib.Util import Util
from PWLib.Campaign import Campaign


def iter_seller_info(seller):

    marketplace = 'US'
    account_name= "vlst"
    account_key= "input the VL blob API key here"
    c = Campaign(marketplace,account_name,account_key)
    print(f"[{current_process().pid}] Getting data for seller {seller} ...")
    try:
        seller_info_dict = {}
        seller_keyw_list = c.GetSellerKeywords(seller)
        seller_asin_list = c.GetSellerASINs(seller)
        seller_info_dict['seller'] = seller
        seller_info_dict['num_of_keywords'] = len(seller_keyw_list)
        seller_info_dict['num_of_asins'] = len(seller_asin_list)
    except:
        seller_info_dict['seller'] = seller
        seller_info_dict['num_of_keywords'] = np.NaN
        seller_info_dict['num_of_asins'] = np.NaN
    return seller_info_dict

def main():
    account_name= "vlst"
    account_key= ""

    marketplace = 'US'
    account_name= "vlst"
    account_key= ""
    c = Campaign(marketplace,account_name,account_key)
    t0 = time.time()
    our_seller_list = c.GetSellerList()

    n_jobs = 10    # You can change this based on your machine

    with Pool(n_jobs) as p:
        results = p.map(iter_seller_info, our_seller_list)

    result_df = pd.DataFrame(results)

    result_df.to_csv(r'C:\Users\Zhaoyu Kou\Desktop\vl_code\seller_camp_info.csv', index=False)

    t1 = time.time()
    total = t1-t0
    print(total)

if __name__ == '__main__':
    main()
