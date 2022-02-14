
from lxml import html
import requests
import numpy as np
import pandas as pd
import datetime
from datetime import date

# ======================================================================================================================
# helpful functions

# generates ERDDAP compatable date
def gen_erddap_date(edate):
    erdate = (str(edate.year) + "-"
              + str(edate.month).zfill(2) + '-'
              + str(edate.day).zfill(2) + "T"
              + str(edate.hour).zfill(2) + ":"
              + str(edate.minute).zfill(2) + ":"
              + str(edate.second).zfill(2) + "Z")

    return erdate


# generates datetime.datetime object from ERDDAP compatable date
def from_erddap_date(edate):

    if isinstance(edate, str):
        temp = datetime.datetime.strptime(str(edate)[:-1], "%Y-%m-%dT%H:%M:%S")
    else:
        return np.nan

    return temp

    # try:
    #     redate = datetime.datetime(year=int(edate[:4]),
    #                                month=int(edate[5:7]),
    #                                day=int(edate[8:10]),
    #                                hour=int(edate[11:13]),
    #                                minute=int(edate[14:16]),
    #                                second=int(edate[17:19]))
    # except TypeError:
    #     return np.nan
    #
    # return redate

''' class used for holding useful information about the ERDDAP databases
========================================================================================================================
'''

class Dataset:
    #dataset object,
    #it takes requested data and generates windows and corresponding urls
    #logger.info('New dataset initializing')

    def __init__(self, url, window_start=False, window_end=False, skip_vars=False):
        self.url = url
        self.t_start, self.t_end = self.data_dates()
        self.skip_vars = list()

        if skip_vars:
            self.skip_vars = skip_vars

        self.data, self.vars = self.get_data()

    #opens metadata page and returns start and end datestamps
    def data_dates(self):
        page = (requests.get(self.url[:-3] + "das")).text

        indx = page.find('Float64 actual_range')
        mdx = page.find(',', indx)
        endx = page.find(";", mdx)
        start_time = datetime.datetime.utcfromtimestamp(float(page[(indx + 21):mdx]))
        end_time = datetime.datetime.utcfromtimestamp(float(page[(mdx + 2):endx]))

        #prevents dashboard from trying to read data from THE FUTURE!
        if end_time > datetime.datetime.now():
            end_time = datetime.datetime.now()

        return start_time, end_time


    def get_data(self):

        self.data = pd.read_csv(self.url, skiprows=[1])
        dat_vars = self.data.columns

        # for set in list(data.keys()):
        self.vars = []
        for var in list(dat_vars):
            if self.skip_vars:
                if var in self.skip_vars:
                    continue

            self.vars.append({'label': var, 'value': var.lower()})

        vars_lower = [each_str.lower() for each_str in dat_vars]

        if 'nerrors' in vars_lower:
            self.vars.append({'label': 'Errors Per Day', 'value': 'errs_per_day'})
        if 'ntrips' in vars_lower:
            self.vars.append({'label': 'Trips Per Day', 'value': 'trips_per_day'})
        if 'sb_depth' in vars_lower:
            self.vars.append({'label': 'Sci Profiles Per Day', 'value': 'sci_profs'})

        self.data.columns = self.data.columns.str.lower()

        if 'dir' in list(self.data.columns):
            if not self.data[self.data['dir'] == 'F'].empty:

                self.vars.append({'label': 'Failures', 'value': 'failures'})
                self.vars.append({'label': 'Time to Failure', 'value': 'time_to_fail'})

        self.data['time'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
        #breakpoint()
        self.data['datetime'] = self.data['time'].apply(from_erddap_date)
        self.data.drop(self.data[self.data['datetime'] > datetime.datetime.today()].index, axis='rows')

        return self.data, self.vars

    def ret_data(self, w_start, w_end):

        #self.data['datetime'] = self.data.loc[:, 'time'].apply(from_erddap_date)

        return self.data[(w_start <= self.data['datetime']) & (self.data['datetime'] <= w_end)]

    def ret_vars(self):

        return self.vars

    def trips_per_day(self, w_start, w_end):

        internal_set =self.data[(w_start <= self.data['datetime']) & (self.data['datetime'] <= w_end)]
        #internal_set['datetime'] = internal_set.loc[:, 'time'].apply(from_erddap_date)
        internal_set['days'] = internal_set.loc[:, 'datetime'].dt.date
        new_df = pd.DataFrame((internal_set.groupby('days')['ntrips'].last()).diff())[1:]
        new_df['days'] = new_df.index

        return new_df

    def errs_per_day(self, w_start, w_end):

        internal_set = self.data[(w_start <= self.data['datetime']) & (self.data['datetime'] <= w_end)]
        #internal_set['datetime'] = internal_set.loc[:, 'time'].apply(from_erddap_date)
        internal_set['days'] = internal_set.loc[:, 'datetime'].dt.date
        new_df = pd.DataFrame((internal_set.groupby('days')['nerrors'].last()).diff())[1:]
        new_df['days'] = new_df.index

        return new_df

    def gen_fail_set(self):

        fail_set = self.data[self.data['dir'] == 'F']

        #fail_set['datetime'] = fail_set.loc[:, 'time'].apply(from_erddap_date)
        fail_set['days'] = fail_set.loc[:, 'datetime'].dt.date
        fail_set = pd.DataFrame((fail_set.groupby('days')['dir'].last()).diff())[1:]
        fail_set['days'] = fail_set.index

        return fail_set

    def sci_profiles_per_day(self, w_start, w_end):

        sci_set = self.data[self.data.loc[:, 'sb_depth'].diff() < -35]
        sci_set['ntrips'] = sci_set['sb_depth'].diff()
        #sci_set['datetime'] = sci_set.loc[:, 'time'].apply(from_erddap_date)
        sci_set['days'] = sci_set.loc[:, 'datetime'].dt.date
        sci_set = pd.DataFrame((sci_set.groupby('days')['ntrips'].size()))
        sci_set['days'] = sci_set.index

        return sci_set