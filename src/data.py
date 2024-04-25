from typing import TypedDict

Data3Dict = TypedDict(
    'Data3Dict', 
    {
        'prod_time': float,
        'setup_time': float,
        'failure_on_product': int,
        'N': int,
        'logfile_path': str
    }
)

data1 = {
    'tc': 31,
    'ts': 57,
    'tw': 109,
}
data2 = {
    'N': 39,
    'tc': 121,
    'ts': 42,
}
data3: Data3Dict = {
    'prod_time': 31.0,
    'setup_time': 32.0,
    'failure_on_product': 10,
    'N': 29,
    'logfile_path': '../media/log.txt',
}