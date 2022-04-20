'''
ERDDAP reader archetype file.

TODO:
    Need exception handler for bad urls

'''

import pandas as pd
import datetime
import requests


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

    if pd.isna(edate):
        return edate

    redate = datetime.datetime(year=int(edate[:4]),
                               month=int(edate[5:7]),
                               day=int(edate[8:10]),
                               hour=int(edate[11:13]),
                               minute=int(edate[14:16]),
                               second=int(edate[17:19]))

    return redate

def erddap_url_date(in_date):
    # generates a url selection date for ERDDAP
    # in the event a datetime.date is passed, it will return midnight (00:00:00) of the date

    if isinstance(in_date, str):

        in_date = from_erddap_date(in_date)

    str_date = f'{in_date.year}-{in_date.month}-{in_date.day}'

    try:
        str_date = f'{str_date}T{in_date.month}%3A{in_date.minute}%3A{in_date.second}Z'
    except AttributeError:
        str_date = f'{str_date}T00%3A00%3A00Z'

    return str_date



''' class used for holding useful information about the ERDDAP databases
========================================================================================================================
'''

class Dataset:
    '''
    dataset object,
    Requires url on init
    If time frames are provided, it will only load set data with those dates

    '''

    def __init__(self, url, window_start=False, window_end=False):
        #self.url = self.url_check(url)
        self.url = url
        self.variables = self.get_raw_vars()
        self.data = pd.DataFrame()
        self.time_flag = False

        if window_start:
            self.t_start = window_start
            self.time_flag = True

        else:
            self.t_start = self.data_start()

        if window_end:
            self.t_end = window_end
            self.time_flag = True

        else:
            self.t_end = self.data_end()

    # def url_check(self, try_url):
    #
    #     page = (requests.get(try_url[:-3] + "das"))

    #reads variables from .das file
    def get_raw_vars(self):
        '''
        get_vars:
        Generates a list of all variables,

        '''

        page = (requests.get(self.url[:-3] + "das")).text
        pages = page.split('\n')

        self.variables = []

        for item in pages:

            if len(item) - len(item.lstrip(' ')) == 2:

                if len(item[2:-2]) > 1:

                    self.variables.append(item[2:-2])


        return self.variables


    #opens metadata page and returns start and end datestamps
    def data_start(self):
        '''
        Let's try using the NC_GLOBAL time_coverage variable, maybe that will be more reliable than the machine
        written dates

        :return:
        '''
        page = (requests.get(self.url[:-3] + "das")).text

        line = page.find('time_coverage_start')
        indx = page.find('"', line)
        endx = page.find('"', indx+1)

        return from_erddap_date(page[indx+1:endx-1])


    def data_end(self):

        '''
        Let's try using the NC_GLOBAL time_coverage variable, maybe that will be more reliable than the machine
        written dates

        :return:
        '''
        page = (requests.get(self.url[:-3] + "das")).text

        line = page.find('time_coverage_end')
        indx = page.find('"', line)
        endx = page.find('"', indx+1)

        return from_erddap_date(page[indx+1:endx-1])


    def get_data(self, time_flag, variables):
        # IDEA:
        # Make both add and exclude variables options with kwargs
        #
        #https://data.pmel.noaa.gov/engineering/erddap/tabledap/TELOM200_PRAWE_M200.csv?time%2Clatitude&time%3E=2022-04-03T00%3A00%3A00Z&time%3C=2022-04-10T14%3A59%3A14Z
        #[url base] + '.csv?time%2C'+ [var1] + '%2C' + [var2] + '%2C' + .... + [time1] + '%3C' + [time2]


        if variables == [] or variables is None:

            self.data = pd.read_csv(self.url, skiprows=[1], low_memory=False)

            return self.data

        spec_url = f'{self.url}'

        if time_flag:
            spec_url = f'{spec_url}?time'
            #spec_url = f'{spec_url}?'

            for var in variables:
                if var is None:
                    continue

                spec_url = f'{spec_url}%2C{var}'

            spec_url = f'{spec_url}&time%3E={erddap_url_date(self.t_start)}&time%3C={erddap_url_date(self.t_end)}'

        else:
            spec_url = f'{spec_url}?{variables[0]}'

            for var in variables[1:]:
                if var is None:
                    continue

                spec_url = f'{spec_url}%2C{var}'

        self.data = pd.read_csv(spec_url, skiprows=[1], low_memory=False)
        temp = self.data['time'].apply(from_erddap_date)
        self.data['time'] = temp

        return self.data


    def ret_windowed_data(self, **kwargs):
        '''
        Returns data and applies time window. The time window may not be necessary with new fncs
        Possibly faster to window via zoom in Plotly app
        '''

        w_start = kwargs.get('t_start', self.t_start)
        w_end = kwargs.get('t_end', self.t_end)

        return self.data[(w_start <= self.data['time']) & (self.data['time'] <= w_end)]


    #converts our basic variable list into Dash compatible dict thingy
    def gen_drop_vars(self, **kwargs):
        '''
        kwargs:
            skips: list
                List of variables to exclude
        :returns list of variables, or list of dict of variales
        '''

        skips = ['time'] + kwargs.get('skips', [])

        vars = []

        for var in list(self.variables):

            # skip unwanted variables
            if var in skips:
                continue

            vars.append({'label': var, 'value': var})

        return vars


    def ret_vars(self):

        #if self

        return self.variables
