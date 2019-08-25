#!/usr/bin/env python
# coding: utf-8

"""
This code is used to preprocess the campaign data to calculate the change of ranks.
For an asin-keyword pair, find the campaigns that last for more than 5 days.
If there is a break for more than 3 days, it will be treated as the end of one campaign.
For a asin-keyword pair, find the campaigns that last for more than 5 days.
Get the V Tracker data for all variation asins, find the most common one, and always try to keep the rank of this most common one. A valid campaign needs to have at least 5 days of the rank record.
Fit a linear regression for the ranks over time, store the slope as the target label for the analysis (a negative slope means the product gained better ranks).
"""


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = 100

import pandas as pd
import sys
import os
import time
from datetime import datetime
from datetime import timedelta
# sys.path.insert(0,'../')
# from importlib import reload
# import PWLib
# reload(PWLib)

from glob import glob

from os import listdir
from os.path import isfile, join


import urllib
# sys.path.insert(0,r'Desktop\vl_code\DataScience')
sys.path.insert(0,'/user/zhaoyu/tree')

import numpy as np
from multiprocessing import Pool, current_process
from importlib import reload
import PWLib as PWLib
reload(PWLib)
from PWLib.VTracker import VTracker

vt_key='Input the V tracker API key here'

reload(PWLib)
from PWLib import AzureBlob
from PWLib import VL_Blobs
from PWLib.VL_Blobs import ASIN
from PWLib.Util import Util
from PWLib.Campaign import Campaign

from sklearn.linear_model import LinearRegression
from sklearn import metrics
import itertools

account_name= "vlst"
account_key= "Input the VL Blob API key here"

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

blob = ASIN(account_name,account_key)
marketplace = 'US'


# In[2]:


def find_var_asins(asin, marketplace = 'US'): # return parent asins and a list of all related asins
    asin_data = blob.GetLatestASINInfo(marketplace=marketplace,asin=asin)
    if 'ParentASIN' in asin_data.keys():
        p_asin = asin_data['ParentASIN']
    else:
        p_asin = asin
    if 'VariationASINs' in asin_data.keys():
        v_asins = asin_data['VariationASINs']
    else:
        v_asins = []
    v_asins.append(p_asin)
    return p_asin, v_asins

########## try to get consecutive dates that meets a criterion
def find_consec_dates(dates, day_tor=2, min_seq=5):
    output_date_list = []
    if len(dates) < min_seq:
        return output_date_list
    else:
        dates_strp = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        date_ints = [d.toordinal() for d in dates_strp]
        ints = np.unique(date_ints)
        int_sorted = np.sort(ints)
        idx_sorted = np.argsort(date_ints)

        diff_list = [1 if (t-s)<=day_tor else 0 for s, t in zip(int_sorted, int_sorted[1:])]
        cum_list = [diff_list[0]]
        for i in range(1,len(diff_list)):
            if diff_list[i-1] == 1 and diff_list[i] !=0:
                cum_list.append(diff_list[i] + cum_list[i-1])
            else:
                cum_list.append(diff_list[i])

        valid_idx_list = []

        for i in range(len(cum_list)):
            if i == len(cum_list)-1:
                if cum_list[i] >= (min_seq-1):
                    output_ls = list(range(i-cum_list[i]+1,i+2))
                    valid_idx_list.append(output_ls)
                else:
                    break
            else:
                if (cum_list[i] >= (min_seq-1)) and (cum_list[i+1] == 0):
                    output_ls = list(range(i-cum_list[i]+1,i+2))
                    valid_idx_list.append(output_ls)

        valid_int_list = [int_sorted[l] for l in valid_idx_list]
        valid_idx_list = [[date_ints.index(i) for i in l] for l in valid_int_list]
        output_date_list = [[dates[i] for i in idx] for idx in valid_idx_list]
        return output_date_list

def most_common(lst): # find the most common element in a list
    return max(set(lst), key=lst.count)

# Get the campaign performance of one seller
def get_seller_campaign(seller, output_path ='./campaign_processed_v3/'):
    try:
        avai_camp_dates = c.GetReportDates(seller=seller)
        date_from = pd.to_datetime(min(avai_camp_dates['Keyword_SponsoredProducts']))
        date_to =   pd.to_datetime(max(avai_camp_dates['Keyword_SponsoredProducts']))
    #     seller_keyw = c.GetSellerKeywords(seller=seller)
        d = c.GetCampaignData(seller=seller,
                                    date_from = date_from,
                                    date_to = date_to,
                                    Get_Asin=True,
                                    Get_ProductAd=False,
                                    Get_Keyword=False,
                                    Get_Campaign=False,
                                    Get_AdGroup=False,
                                    Get_ProductAttributeTargeting=False,

                                    Get_SponsoredProducts=True,
                                    Get_SponsoredBrands=False,

                                    Get_Suffix_Query=False,
                                    Get_Suffix_Placement=False)

        camp_asin_df = d['Asin_SponsoredProducts']
        camp_asin_df = camp_asin_df[camp_asin_df['Report_KeywordText']!='*']
        report_asin_ls = camp_asin_df['Report_Asin'].unique()
        asin_summary_merged = pd.DataFrame()
        slope_ls = []
    #     ctr_counter = 0
        keyw_pasin_buffer = []

        for selected_asin in report_asin_ls:
            # selected_asin = report_asin_ls[0]
            sub_asin_df = camp_asin_df[camp_asin_df['Report_Asin']==selected_asin]
            sub_keyw_ls = sub_asin_df['Report_KeywordText'].unique()
            for selected_keyw in sub_keyw_ls:
        # selected_keyw = sub_keyw_ls[0]
                p_asin, v_asin_list  = find_var_asins(selected_asin)
                keyw_pasin = selected_keyw + p_asin
                if keyw_pasin not in keyw_pasin_buffer:
                    keyw_pasin_buffer.append(keyw_pasin)
                    slt_asin_df = camp_asin_df[camp_asin_df['Report_Asin'].isin(v_asin_list)]
                    slt_keyw_asin_df = slt_asin_df[slt_asin_df['Report_KeywordText']==selected_keyw] # a subset of the asins chosen for this keyword in campains

                    asin_summary_ls = []
                    camp_dates = []
                    for asin in slt_asin_df['Report_Asin'].unique():
                        asin_sum_dict = {}
                        asin_sum_dict['asin'] = asin
                        asin_sum_dict['keyword'] = slt_keyw_asin_df['Report_KeywordText'].unique()[0]
                        sub_df = slt_asin_df[slt_asin_df['Report_Asin']==asin]
                        camp_dates1 = list(sub_df['ReportDate'].unique())
                        asin_sum_dict['campaign_dates'] = camp_dates1
                        camp_dates = camp_dates + camp_dates1
                        asin_sum_dict['is_parent_asin'] = 1 if asin==p_asin else 0
                        asin_sum_dict['parent_asin'] = p_asin
                        asin_sum_dict['seller'] = seller
                        asin_summary_ls.append(asin_sum_dict)

                    asin_summary_df = pd.DataFrame(asin_summary_ls)
                    asin_summary_merged = pd.concat([asin_summary_merged, asin_summary_df], axis=0, ignore_index=True, sort=False)
                    camp_dates = list(set(camp_dates))

                    valid_dates_ls = find_consec_dates(camp_dates)

                    for d_ls in valid_dates_ls:
                        if len(d_ls) != 0:
                            from_date = pd.to_datetime(min(d_ls))
                            to_date =   pd.to_datetime(max(d_ls)) + timedelta(days=1)
                            with VTracker(secret=vt_key) as vt:
                                try:
                                    df_vt = vt.GetVTrackerData(keywords=selected_keyw,
                                                       asins=v_asin_list,
                                                       from_date=from_date,
                                                       to_date=to_date)
                                except:
                                    df_vt = pd.DataFrame()
                            if len(df_vt)>=3:
                                ## remove duplicates based on scrape date, if the most common asin is available, keep the most commone ones
                                ## or keep the one with the highest rank
                                df_vt['date'] = df_vt['scrapeDate'].apply(lambda x:x.split('T')[0])
                                major_asin = most_common(list(df_vt['asin']))
                                df_vt['is_major_asin'] = df_vt['asin'].apply(lambda x: int(x==major_asin))
                                df_vt = df_vt.sort_values(['is_major_asin','rank'], ascending = [False, True]).drop_duplicates(subset='date') # keep the highest rank for a day
                                if len(df_vt)>=5:
                                    df_vt['time'] = pd.to_datetime(df_vt['scrapeDate'])
                                    min_day = min(df_vt['time'])
                                    df_vt['day_diff'] = df_vt['time'].apply(lambda x:(x-min_day).days)
                                    min_rank = min(df_vt['rank'])
                                    df_vt['rank_diff'] = df_vt['rank'].apply(lambda x:(x-min_rank))
                                    # find the slope
                                    X = df_vt['day_diff'].values.reshape(-1,1)
                                    Y = df_vt['rank_diff'].values.reshape(-1,1)
                                    regressor = LinearRegression()
                                    regressor.fit(X, Y)
                                    slope = regressor.coef_[0][0]
                                    df_vt = df_vt.sort_values('date', ascending = True)
                                    out_dict = {}
                                    out_dict['parent_asin'] = p_asin
                                    out_dict['keyword'] = selected_keyw
                                    out_dict['seller'] = seller
                                    out_dict['variation_asins'] = v_asin_list
                                    out_dict['campaign_dates'] = d_ls
                                    out_dict['rank_dates'] = list(np.sort(df_vt['date']))
                                    out_dict['ranks'] = list(df_vt['rank'])
                                    out_dict['record_asin'] = list(df_vt['asin'])
                                    out_dict['num_ads'] = list(df_vt['rankIncludingAds']-df_vt['rank'])
                                    out_dict['slope'] = slope
                                    slope_ls.append(out_dict)
    #                                 ctr_counter += 1
        slope_df = pd.DataFrame(slope_ls)
        file_n_slope = output_path + seller + '/' + seller + '_slope.csv'
        os.makedirs(os.path.dirname(file_n_slope), exist_ok=True)
        slope_df.to_csv(file_n_slope, index = False)
        file_n_summary = output_path + seller + '/' + seller + '_summary.csv'
        os.makedirs(os.path.dirname(file_n_summary), exist_ok=True)
        asin_summary_merged.to_csv(file_n_summary, index = False)
        print(seller)
    except:
        print('Failed: {}'.format(seller))


# In[3]:
if __name__ == '__main__':

    # read in the seller summary
    seller_info = pd.read_csv('seller_camp_info.csv')
    seller_info = seller_info.sort_values(['num_of_asins', 'num_of_keywords'], ascending=[False, False])
    seller_info = seller_info.reset_index(drop=True)

    marketplace = 'US'
    account_name= "vlst"
    account_key= "Input the VL Blob API key here"
    c = Campaign(marketplace,account_name,account_key)

    # existing_seller = [x.split('_')[0] for x in listdir('./campaign_processed/') if "slope" in x]
    # # [x for x in glob("./campaign_processed/") if "slope" in x]
    # len(existing_seller)

    seller_list = list(seller_info['seller'].unique())
    existing_seller = [x.split('/')[-1] for x in listdir('./campaign_processed_v3/')]
    left_seller_list = [s for s in seller_list if s not in existing_seller]

    #
    # left_seller_list = [s for s in seller_list if s not in existing_seller]

    n_jobs = 30    # You can change this based on your machine

    with Pool(n_jobs) as p:
        p.map(get_seller_campaign, left_seller_list)
