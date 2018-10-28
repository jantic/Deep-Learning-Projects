from fastai.structured import *
from fastai.column_data import *

def cleanup_year_column(dataframe, raw_field, clean_field):
    dataframe[clean_field]=dataframe[raw_field].fillna(1900).astype(np.int32)
    dataframe[clean_field]=np.where(dataframe[clean_field] < 1900, 1900, dataframe[clean_field])  
    dataframe.drop(columns=[raw_field], inplace=True)

def cleanup_month_column(dataframe, raw_field, clean_field):
    dataframe[clean_field]=dataframe[raw_field].fillna(1).astype(np.int32)
    dataframe[clean_field]=np.where(dataframe[clean_field] > 12, 1, dataframe[clean_field])   
    dataframe[clean_field]=np.where(dataframe[clean_field] < 1, 1, dataframe[clean_field])   
    dataframe.drop(columns=[raw_field], inplace=True)
    
def cleanup_day_column(dataframe, year_field_clean, month_field_clean, raw_day_field, day_field_clean, datetime_field):
    dataframe[day_field_clean]=dataframe[raw_day_field].fillna(1).astype(np.int32)
    dataframe[day_field_clean]=np.where(dataframe[day_field_clean] > 30, 1, dataframe[day_field_clean])   
    dataframe[day_field_clean]=np.where(dataframe[day_field_clean] < 1, 1, dataframe[day_field_clean]) 
    dataframe.drop(columns=[raw_day_field], inplace=True)
    #Makes sure only valid dates are used, by forcing invalid ones to NaN then fixing the NaN fields accordingly
    #afterwards
    generate_datetime_column(dataframe=dataframe, year_field=year_field_clean, month_field=month_field_clean, 
                             day_field=day_field_clean, datetime_field=datetime_field, errors_to_na=True)
    #Convert invalid datetime days to 1
    dataframe[day_field_clean]=np.where(dataframe[datetime_field].isnull(), 1, dataframe[day_field_clean])   
    
def generate_datetime_column(dataframe, year_field, month_field, day_field, datetime_field, errors_to_na=False):
    errors = 'coerce' if errors_to_na else 'raise'
    dataframe[datetime_field]=pd.to_datetime(dict(year=dataframe[year_field], month=dataframe[month_field], 
                                                      day=dataframe[day_field]), errors='coerce')

#Fills in missing data with default fields; corrects for invalid dates; constructs a new datetime column; generates
#extra date information (like day of week, day of year, etc); and removes the old date columns as they're no longer needed
def process_dates(dataframe, year_field, month_field, day_field, datetime_field):
    clean_suffix = '_clean'
    year_field_clean = year_field + clean_suffix
    month_field_clean =month_field + clean_suffix
    day_field_clean = day_field + clean_suffix
    
    cleanup_year_column(dataframe=dataframe, raw_field=year_field, clean_field=year_field_clean)
    cleanup_month_column(dataframe=dataframe, raw_field=month_field, clean_field=month_field_clean)
    cleanup_day_column(dataframe=dataframe, year_field_clean=year_field_clean, month_field_clean=month_field_clean, 
                       raw_day_field=day_field, day_field_clean=day_field_clean, datetime_field=datetime_field)
    generate_datetime_column(dataframe=dataframe, year_field=year_field_clean, month_field=month_field_clean, 
                         day_field=day_field_clean, datetime_field=datetime_field, errors_to_na=False)   
    add_datepart(df=dataframe, fldname=datetime_field, drop=False)