import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

# engagement = pd.read_csv(r'D:\Source\experiments\Data science & AI\Basic inferential statistics\engagement_ratio_data.csv', header=None)
# engagement = engagement.iloc[:,0]
# engagement.std()
# engagement.describe()

# sample = np.array([8, 9, 12, 13, 14, 16])
# pop_std = 2.8
# sample.mean() # 12

# print(12 - 1.96 * (2.8/(6 ** .5)))
# print(12 + 1.96 * (2.8/(6 ** .5)))

# st.norm.ppf(.999)

# print(2.41/30**.5) # 0.44000378786248345
# print((8.3-7.47)/0.44000378786248345)

# print(2.41/50**.5) # 0.3408254685319159
# print((8.3-7.47)/0.3408254685319159)

# print(230/5**.5) # 102.85912696499032
# print((9640-7895)/102.85912696499032) # 16.964950524944058

''' Lesson 6 | t-Tests Part 1 '''

def df_from_google_sheet_url(url):
    export_attr = '/export?gid=0&format=csv'
    df = pd.read_csv('{}{}'.format(url, export_attr))
    return df

def plot_t_dist(degrees_of_freedom):

    # Define default linear space (x)
    norm_linspace = np.linspace(\
        #st.t.ppf(0.0001, degrees_of_freedom),\
        st.norm.ppf(0.0001),\
        #st.t.ppf(0.9999, degrees_of_freedom),\
        st.norm.ppf(0.9999),\
        100\
    )

    # Plot normal distribuition for comparison 
    plt.plot(\
        norm_linspace, \
        st.norm.pdf(norm_linspace), \
        label='Normal', color='gray', alpha=.5, linestyle='--' \
    )

    # Plot t distribuition 
    plt.plot(\
        norm_linspace, \
        st.t.pdf(norm_linspace, degrees_of_freedom), \
        label='t dist. w/ {} df'.format(degrees_of_freedom) \
    )

    plt.legend(loc='upper left')

def lesson_6_quiz_16():
    '''
    Quiz 16 | n and DF
    '''
    # avg_beak_width = 6.07
    df_from_google_sheet_url(
        r'https://docs.google.com/spreadsheets/d/1PMtYHFuOAkKJGQTx5tsqjvbPSbA_kRBKlCD6KEqSAXk'
    )

def lesson_6_quiz_20():
    '''
    Quiz 20 | P-value
    '''
    base = 10
    sample = pd.Series([5, 19, 11, 23, 12, 7, 3, 21])

    # compute t value
    t = (sample.mean() - base) / (sample.std() / (sample.size ** .5))
    # 0.977461894333816

    # prob. of getting a value < t
    p = st.t.cdf(t, df=sample.size-1) 
    # 0.8215155024706349

    # prob. of getting a value >= t
    p = 1 - p
    # 0.17848449752936513

    # mult. by 2 since its a 2-tailed test
    p *= 2
    # 0.35696899505873025
    print(p)

def lesson_6_quiz_23():
    '''
    Quiz 23 | t-Critical Values
    '''
    # Ho: µ = 1830
    # Ha: µ ≠ 1830

    mean_rent = 1830
    M = 1700
    S = 200
    n = 25
    alpha = .05

    t_crit = st.t.ppf(1 - (alpha / 2.0), df=n-1)
    # ±2.0638985616280205

    t_crit = round(t_crit, 4)
    # rounding to 2.064 to match t-table
    
    t = (M - mean_rent) / (S / (n ** .5))
    # -3.25

    if abs(t) > t_crit:
        print('Null rejected (|{}| > {})'.format(t, t_crit))
    else:
        print('Null retained (|{}| <= {})'.format(t, t_crit))
    ## Null rejected (|-3.25| > 2.064), µ ≠ 1830

    plot_t_dist(n-1)
    
    d = (M - mean_rent) / S
    print('d = {}'.format(d))
    # -0.65

    # 2.064(200/sqrt(25))
    margin_of_error = t_crit * (S / (n ** .5))
    print(margin_of_error)
    #  1700 ± margin_of_error
    ci = (M - margin_of_error, M + margin_of_error)
    # (1617.444, 1782.556)
    print('Confidence interval = {}'.format(ci))

def lesson_6_quiz_32():
    '''
    Quiz 32 | Keyboards
    '''
    keyboard_data = df_from_google_sheet_url(
        r'https://docs.google.com/spreadsheets/d/1rTzt0J53tYRwE3d_-WfZ0t4aok3uhc-SP_3G83iSGFg'
    )
    keyboard_data.describe()
    qwerty_errors = keyboard_data.iloc[:,0]
    abcdef_errors = keyboard_data.iloc[:,1]

    n = qwerty_errors.size

    # Ho: µ(qwerty) = µ(abcdef)     same as     µ(qwerty) - µ(abcdef) = 0
    # Ha: µ(qwerty) ≠ µ(abcdef)

    point_estimate = mean_diff = qwerty_errors.mean() - abcdef_errors.mean()
    # -2.7199999999999998

    S = (qwerty_errors - abcdef_errors).std()
    # 3.6914315199752337

    t = mean_diff / ( S / n ** .5)
    # -3.6842075835369266

    alpha = .05

    t_critical_value = st.t.ppf(1 - (alpha / 2.0), df=n-1)
    # ±2.0638985616280205

    if abs(t) > t_critical_value:
        print('Null rejected (|{}| > {})'.format(t, t_critical_value))
    else:
        print('Null retained (|{}| <= {})'.format(t, t_critical_value))
    ## Null rejected (|-3.684| > 2.064), µ(qwerty) ≠ µ(abcdef)

    d = mean_diff / S
    # -0.7368415167073853

    critical_region = t_critical_value * (S / (n ** .5))
    # -2.7199999999999998
    ci = (mean_diff - critical_region, mean_diff + critical_region)
    # (-4.243748040885044, -1.1962519591149554)
    
    print('Confidence interval = {}'.format(ci))
    
    # Students will make on average around 4 to 1 fewer typos on the qwerty

''' Lesson 7 | Problem Set 10a '''

# t_crit = st.t.ppf(.95, df=999)
# Sd = ((1.2**2)+(2.7**2))**.5 = (9-0)/(Sd/(1000**.5)))

''' Lesson 8 | t-Tests Part 2 '''

# t_05 = st.t.ppf(.05, df=24) # -1.7108820799094282
# t = -25/10 # -2.5
# p = st.t.cdf(-2.5, df=24) # 0.009827087558289377
# d = -25/50 # -0.5
# r2 = t**2/((t**2)+24) # 0.2066115702479339

# t_025 = st.t.ppf(.025, df=24) # -2.063898561628021
# margin_of_error = abs(t_025 * 10) # 20.63898561628021
# CI = (126-margin_of_error, 126+margin_of_error) # (105.36101438371979, 146.6389856162802)