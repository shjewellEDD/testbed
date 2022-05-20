'''
ERDDAP reader archetype file.

TODO:
    Need exception handler for bad urls
    Do we need the Prawler specific functions, or should those be offloaded
        Prawer specific fuctions should have date guards added
    Maybe make ret_data invoke get_date if self.data is empty?

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

    elif len(edate) > 10:

        redate = datetime.datetime(year=int(edate[:4]),
                                   month=int(edate[5:7]),
                                   day=int(edate[8:10]),
                                   hour=int(edate[11:13]),
                                   minute=int(edate[14:16]),
                                   second=int(edate[17:19]))

    else:
        redate = datetime.datetime(year=int(edate[:4]),
                                   month=int(edate[5:7]),
                                   day=int(edate[8:10]),
                                   hour=0,
                                   minute=0,
                                   second=0)

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



''' 
class used for holding useful information about the ERDDAP databases
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
        self.raw_vars = self.get_raw_vars()
        self.data = pd.DataFrame()
        self.window_flag = False
        self.time_flag = False
        
        if 'time' in self.raw_vars:

            self.time_flag = True

            if window_start:
                self.t_start = window_start
                self.window_flag = True
    
            else:
                self.t_start = self.data_start()
    
            if window_end:
                self.t_end = window_end
                self.window_flag = True
    
            else:
                self.t_end = self.data_end()
        

    def date_guard(self, in_date):
        '''
        Our datasets love to send us data from the future. Let's stop that, okay?
        :param in_date:
        :return: datetime.datetime
        '''

        if in_date > datetime.datetime.today():

            return datetime.datetime.today()

        return in_date

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

        self.raw_vars = []

        for item in pages:

            if len(item) - len(item.lstrip(' ')) == 2:

                if len(item[2:-2]) > 1:

                    self.raw_vars.append(item[2:-2])


        return self.raw_vars


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

        return self.date_guard(from_erddap_date(page[indx+1:endx-1]))


    def get_data(self, **kwargs):
        # IDEA:
        # Make both add and exclude variables options with kwargs
        #
        #https://data.pmel.noaa.gov/engineering/erddap/tabledap/TELOM200_PRAWE_M200.csv?time%2Clatitude&time%3E=2022-04-03T00%3A00%3A00Z&time%3C=2022-04-10T14%3A59%3A14Z
        #[url base] + '.csv?time%2C'+ [var1] + '%2C' + [var2] + '%2C' + .... + [time1] + '%3C' + [time2]

        self.data = pd.DataFrame()

        variables = kwargs.get('variables', self.raw_vars)

        self.t_start = kwargs.get('window_start', self.t_start)
        self.t_end = kwargs.get('window_end', self.t_end)

        if 'window_start' in kwargs or 'window_end' in kwargs:
            self.window_flag = True

        vars = set(variables)

        spec_url = f'{self.url}'

        # duplicate vars cause HTML causes errors, sets don't have duplicates

        if self.time_flag:
            spec_url = f'{spec_url}?time'
            #spec_url = f'{spec_url}?'

            for var in vars:
                if var is None:
                    continue

                if var == 'time':
                    continue

                if var == 'NC_GLOBAL':
                    continue

                # don't try to load non-existant variables, 'kay?
                if var not in self.raw_vars:
                    continue

                spec_url = f'{spec_url}%2C{var}'

            spec_url = f'{spec_url}&time%3E={erddap_url_date(self.t_start)}&time%3C={erddap_url_date(self.t_end)}'

        else:
            spec_url = f'{spec_url}?'

            for var in variables:
                if var is None:
                    continue

                if var == 'time':
                    continue

                if var == 'NC_GLOBAL':
                    continue

                spec_url = f'{spec_url}%2C{var}'

        self.data = pd.read_csv(spec_url, skiprows=[1], low_memory=False)

        if self.time_flag:
            temp = self.data['time'].apply(from_erddap_date)
            self.data['time'] = temp

        return self.data


    def ret_data(self, **kwargs):
        '''
        Returns data and applies time window. The time window may not be necessary with new fncs
        Possibly faster to window via zoom in Plotly app
        '''

        if self.time_flag:

            w_start = kwargs.get('t_start', self.t_start)
            w_end = self.date_guard(kwargs.get('t_end', self.t_end))

            return self.data[(w_start <= self.data['time']) & (self.data['time'] <= w_end)]

        else:

            return self.data


    #converts our basic variable list into Dash compatible dict thingy
    def gen_drop_vars(self, **kwargs):
        '''
        kwargs:
            skips: list
                List of variables to exclude
        :returns list of dict of variales, ERDDAP compatible
        '''

        skips = ['time', 'NC_GLOBAL'] + kwargs.get('skips', [])

        vars = []

        for var in list(self.raw_vars):

            # skip unwanted variables
            if var in skips:
                continue

            vars.append({'label': var, 'value': var})

        return vars


    def ret_raw_vars(self):
        '''
        Returns variables with none skipped
        :return: list
        '''

        return self.raw_vars

    def ret_vars(self, **kwargs):
        '''
        Return
        :param kwargs: keyword: skips, list of strs, variables to skip
        :return:
        '''

        skips = ['time', 'latitude', 'longitude', 'timeseries_id']

        external_skips = kwargs.get('skips', None)

        if external_skips:

            skips = skips + external_skips

        vars = []

        for var in list(self.raw_vars):

            # skip unwanted variables
            if var in skips:
                continue

            vars.append(var)

        return vars

    def trips_per_day(self, w_start, w_end):
        '''
        Prawler specific function, calculates round trips per day
        :param w_start:
        :param w_end:
        :return:
        '''

        internal_set =self.data[(w_start <= self.data['datetime']) & (self.data['datetime'] <= w_end)]
        #internal_set['datetime'] = internal_set.loc[:, 'time'].apply(from_erddap_date)
        internal_set['days'] = internal_set.loc[:, 'datetime'].dt.date
        new_df = pd.DataFrame((internal_set.groupby('days')['ntrips'].last()).diff())[1:]
        new_df['days'] = new_df.index

        return new_df

    def errs_per_day(self, w_start, w_end):
        '''
        Pralwer specific function
        Calculates errors per day
        :param w_start:
        :param w_end:
        :return:
        '''

        internal_set = self.data[(w_start <= self.data['datetime']) & (self.data['datetime'] <= w_end)]
        # internal_set['datetime'] = internal_set.loc[:, 'time'].apply(from_erddap_date)
        internal_set['days'] = internal_set.loc[:, 'datetime'].dt.date
        new_df = pd.DataFrame((internal_set.groupby('days')['nerrors'].last()).diff())[1:]
        new_df['days'] = new_df.index

        return new_df

    def gen_fail_set(self):
        '''
        Prawler specific function
        Returns fails
        :return:
        '''

        fail_set = self.data[self.data['dir'] == 'F']
        # fail_set['datetime'] = fail_set.loc[:, 'time'].apply(from_erddap_date)
        fail_set['days'] = fail_set.loc[:, 'datetime'].dt.date
        fail_set = pd.DataFrame((fail_set.groupby('days')['dir'].last()).diff())[1:]
        fail_set['days'] = fail_set.index

        return fail_set

    def sci_profiles_per_day(self, w_start, w_end):
        '''
        Prawler specific function, returns the number of scientific profiles per day
        :param w_start:
        :param w_end:
        :return:
        '''

        sci_set = self.data[self.data.loc[:, 'sb_depth'].diff() < -35]
        sci_set['ntrips'] = sci_set['sb_depth'].diff()

        # sci_set['datetime'] = sci_set.loc[:, 'time'].apply(from_erddap_date)
        sci_set['days'] = sci_set.loc[:, 'datetime'].dt.date
        sci_set = pd.DataFrame((sci_set.groupby('days')['ntrips'].size()))
        sci_set['days'] = sci_set.index

        return sci_set
