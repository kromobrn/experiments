import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO


SHEET_URL = r'https://docs.google.com/spreadsheets/d/15rIW21lVrPTzAS1EcqPGDz1pPnCgMBZ2HVS-0iBtUFA/edit#gid=0'
TARGET_HOURS = 42.5
BALANCE = 0

def df_from_google_sheet(url):
    csv_export_url = url.replace('/edit#gid=', '/export?format=csv&gid=')
    return pd.read_csv(csv_export_url)

df = df_from_google_sheet(SHEET_URL)
df.set_index('Data')


days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
