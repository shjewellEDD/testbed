
import pandas as pd
import datetime
import requests


skipvars = ['latitude', 'longitude']
# will need to be altered for multi-set displays

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

''' class used for holding useful information about the ERDDAP databases
========================================================================================================================
'''

class Dataset:
    #dataset object,
    #it takes requested data and generates windows and corresponding urls
    #logger.info('New dataset initializing')

    def __init__(self, url, window_start=False, window_end=False):
        self.url = url
        self.flags = pd.DataFrame
        self.data, self.vars = self.get_data()
        self.t_start, self.t_end = self.data_dates()
        self.set_names, self.flags = self.catagorize()

        #self.co2_vars

    #opens metadata page and returns start and end datestamps
    def data_dates(self):
        '''
        Currently the meta data states a start in 1969, which seems unrealistic
        Let's actually use hard coded dates. We may need to fix
        :return:
        '''
        page = (requests.get(self.url[:-3] + "das")).text

        indx = page.find('Float64 actual_range')
        mdx = page.find(',', indx)
        endx = page.find(";", mdx)
        # start_time = datetime.datetime.utcfromtimestamp(float(page[(indx + 21):mdx]))
        # end_time = datetime.datetime.utcfromtimestamp(float(page[(mdx + 2):endx]))

        #prevents dashboard from trying to read data from ... THE FUTURE!
        # if end_time > datetime.datetime.now():
        #     end_time = datetime.datetime.now()

        if self.data['time'].min() < datetime.datetime.utcfromtimestamp(float(page[(indx + 21):mdx])):
            start_time = datetime.datetime.utcfromtimestamp(float(page[(indx + 21):mdx]))
        else:
            start_time = self.data['time'].min()

        if self.data['time'].max() > datetime.datetime.utcfromtimestamp(float(page[(mdx + 2):endx])):
            end_time = self.data['time'].max()
        else:
            end_time = datetime.datetime.utcfromtimestamp(float(page[(mdx + 2):endx]))

        return start_time, end_time


    def get_data(self):

        self.data = pd.read_csv(self.url, skiprows=[1], low_memory=False)
        temp = self.data['time'].apply(from_erddap_date)
        #self.serials = (self.data['SN_ASVCO2'].unique()).tolist()
        #self.data = self.data.select_dtypes(include='float64')
        self.data['time'] = temp
        self.flags.assign(temp)

        dat_vars = self.data.columns


        # for set in list(data.keys()):
        self.vars = []
        for var in list(dat_vars):
            # if var in skipvars:
            #     continue

            if 'FLAG' in var:
                self.flags.assign(self.data[var])
                #self.data.drop(self.data[var], axis='columns')

            if str(self.data[var].dtype) == 'object':
                continue

            self.vars.append({'label': var, 'value': var})


        return self.data, self.vars


    def catagorize(self):

        sets = set()
        flags = pd.DataFrame
        base_sets = {}

        subsets = {0:   '_MEAN',
                   1:   '_STDDEV',
                   2:   '_MAX',
                   3:   '_MIN'
                   }

        for col in self.data.columns:

            for n in subsets:

                if "FLAG" in col:

                    flags.assign(self.data[col])
                    continue

                elif subsets[n] in col:

                    sets.add(col.replace(subsets[n], ''))

                    try:
                        base_sets[col.replace(subsets[n], '')][subsets[n]] = col
                    except KeyError:
                        base_sets[col.replace(subsets[n], '')] = {subsets[n]: col}


                    continue

            #drop_list =

        return base_sets, flags

    def ret_data(self, **kwargs):

        w_start = kwargs.get('t_start', self.t_start)
        w_end = kwargs.get('t_end', self.t_end)

        #self.data['datetime'] = self.data.loc[:, 'time'].apply(from_erddap_date)

        return self.data[(w_start <= self.data['time']) & (self.data['time'] <= w_end)]

    def ret_vars(self):

        return self.vars
