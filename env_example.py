#read only credentials
host = '000.000.000.000'
username = 'username'
user = username
password = 'password'

def get_db_url(db_url):
    url = f'mysql+pymysql://{user}:{password}@{host}/{db_url}'
    return url