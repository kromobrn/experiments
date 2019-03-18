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

def dropna(ndarray):
    return ndarray[~np.isnan(ndarray)]

def sst(samples):
    '''
    Calculates the Sum of Squares of Treatment/Group/Samples (Between)
    '''
    gmean = dropna(samples.flatten()).mean()

    def squared_deviation_from_gmean(sample):
        return (dropna(sample).mean() - gmean) ** 2

    def func(sample):
        s = dropna(sample)
        return s.size * squared_deviation_from_gmean(s)
    
    return np.apply_along_axis(func, axis=1, arr=samples).sum()

def sse(samples):
    '''
    Calculates the Sum of Squares of Error (Within)
    '''
    def squared_deviation_from_mean(subject, sample_mean):
        return (subject - sample_mean) ** 2

    def func(sample):
        s = dropna(sample)
        sum_sqrs = np.apply_along_axis(
            squared_deviation_from_mean, 0, s, s.mean()
        )
        ret = np.zeros(sample.size)
        ret[:s.size] = sum_sqrs
        return ret

    return np.apply_along_axis(func, axis=1, arr=samples).sum()

def dft(samples):
    return samples.shape[0] - 1

def dfe(samples):
    return dropna(samples).size - samples.shape[0]

def mst(samples):
    return sst(samples) / dft(samples)

def mse(samples):
    return sse(samples) / dfe(samples)

def f_statistic(samples):
    return mst(samples) / mse(samples)

def linear_prediction(x, sample_x, sample_y):
    r, _ = st.pearsonr(sample_x, sample_y)
    slope = r * sample_y.std(ddof=1) / sample_x.std(ddof=1)
    intercept = sample_y.mean() - slope * sample_x.mean()
    return slope * x + intercept

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

def lesson_7_quiz_X():
    ''' 
    Lesson 7 | Problem Set 10a 
    '''

    # t_crit = st.t.ppf(.95, df=999)
    # Sd = ((1.2**2)+(2.7**2))**.5 = (9-0)/(Sd/(1000**.5)))
    pass

def lesson_8_quiz_X():
    ''' 
    Lesson 8 | t-Tests Part 2 
    '''

    # t_05 = st.t.ppf(.05, df=24) # -1.7108820799094282
    # t = -25/10 # -2.5
    # p = st.t.cdf(-2.5, df=24) # 0.009827087558289377
    # d = -25/50 # -0.5
    # r2 = t**2/((t**2)+24) # 0.2066115702479339

    # t_025 = st.t.ppf(.025, df=24) # -2.063898561628021
    # margin_of_error = abs(t_025 * 10) # 20.63898561628021
    # CI = (126-margin_of_error, 126+margin_of_error) # (105.36101438371979, 146.6389856162802)
    pass

def lesson_9_quiz_X():
    pre = np.array([8, 7, 6, 9, 10, 5, 7, 11, 8, 7])
    pos = np.array([5, 6, 4, 6, 5, 3, 2, 9, 4, 4])
    diff = pos - pre

    tc = st.t.ppf(.05, df=9)

    se = diff.std(ddof=1) / diff.size ** .5

    t = diff.mean() / se
    d = diff.mean() / diff.std(ddof=1)

    tc = st.t.ppf(.025, df=9)
    margin_of_error = abs(tc * se)
    ci = (diff.mean() - margin_of_error, diff.mean() + margin_of_error)
    r2 = (t ** 2) / ((t ** 2) + 9)

def lesson_10_quiz_3():
    ''' 
    Quiz 3 | Meal Prices
    '''
    # Ho: µ(gettysburg) = µ(wilma)
    # Ha: µ(gettysburg) ≠ µ(wilma)

    sheet_url = r'https://docs.google.com/spreadsheets/d/1bNeiNwHKgTbg-6h_ahFuDUd_TkVWDuIrTvELAvcKzUc'
    df = df_from_google_sheet_url(sheet_url)

    gettysburg_data, wilma_data = df.iloc[:,0], df.iloc[:,1]
    gettysburg_data.dropna(inplace=True)
    wilma_data.dropna(inplace=True)

    gettysburg_data.describe() # mean = 8.944444, std = 2.645134
    wilma_data.describe() # mean = 11.142857, std = 2.178819

    se = (gettysburg_data.var() / gettysburg_data.size
            + wilma_data.var() / wilma_data.size) ** .5
    # 0.8531100847677228

    mean_diff = gettysburg_data.mean() - wilma_data.mean() # -2.1984126984126977
    t = mean_diff / se # -2.576939058235681

    alpha = .05
    two_tailed = True
    df = gettysburg_data.size + wilma_data.size - 2

    # In a two tailed test the probability is split between both extremes
    prob = alpha if not two_tailed else alpha * .5

    t_crit = st.t.ppf(prob, df=df)
    # -2.042272456301238

    # |-2.576939058235681| > |-2.042272456301238| Null rejected

    # Probability of getting a sample under the lower (left) critical region
    p = st.t.cdf(t, df=df)

    p = p if not two_tailed else p * 2
    # 0.01512946515275134
    print('alpha={}, {}-tailed'.format(alpha, 'two' if two_tailed else 'one'))
    print('t({})={}, p={}'.format(df, t, p))
    pass

def lesson_10_quiz_12():
    ''' 
    Quiz 12 | Acne Medication
    '''
    # Ho: µ(a) = µ(b)
    # Ha: µ(a) ≠ µ(b)

    drug_a = pd.Series([.4, .36, .2, .32, .45, .28])
    drug_b = pd.Series([.41, .39, .18, .23, .35])

    two_tailed = True
    alpha = .05
    df = drug_a.size + drug_b.size - 2

    mean_diff = drug_a.mean() - drug_b.mean()
    se = (drug_a.var() / drug_a.size
        + drug_b.var() / drug_b.size) ** .5

    t = mean_diff / se

    prob = alpha if not two_tailed else alpha * .5
    t_critical = st.t.ppf(prob, df=df)

    p = 1 - st.t.cdf(t, df=df)
    p = p if not two_tailed else p * 2

    print('alpha={}, {}-tailed'.format(alpha, 'two' if two_tailed else 'one'))
    print('t({})={}, p={}'.format(df, t, p))

    # Null retained (|0.39547554497329196| <= |-2.262157162740992|)
    # The two drugs have not a significant difference in ther effects on acne
    pass

def lesson_10_quiz_16():
    ''' 
    Quiz 16 | Who Has More Shoes
    '''
    # Ho: µ(f) = µ(m)    --    µ(f) - µ(m) = 0
    # Ha: µ(f) ≠ µ(m)

    females = pd.Series([90, 28, 30, 10, 5, 9, 60])
    males = pd.Series([4, 120, 5, 3, 10, 3, 5, 13, 4, 10, 21])

    females.describe()
    males.describe()

    two_tailed = True
    alpha = .05
    df = females.size + males.size - 2

    mean_diff = females.mean() - males.mean()
    se = (females.var() / females.size
            + males.var() / males.size) ** .5

    t = mean_diff / se

    prob = alpha if not two_tailed else alpha * .5
    t_critical = st.t.ppf(prob, df=df)

    # Null retained (|0.9629743503795974| < |-2.1199052992210112|)
    # No significant difference in the number of pairs of shoes owned by females and males

    margin_of_error = abs(t_critical) * se
    ci = (mean_diff - margin_of_error, mean_diff + margin_of_error)
    # (-18.192841871177293, 48.478556156891585)
    # 95% interval for the TRUE difference between pairs of shoes owned by females and males

    r2 = (t ** 2) / ((t ** 2) + df)
    # 0.05478242400037163
    # Only about 5% of the difference in pairs of shoes ownred can be attributed to gender
    pass

def lesson_10_quiz_23():
    '''
    Quiz 23 | Pooled Variance
    '''
    x = pd.Series([5, 6, 1, -4])
    y = pd.Series([3, 7, 8])

    x_ss = ((x - x.mean()) ** 2).sum()
    y_ss = ((y - y.mean()) ** 2).sum()

    x_df = x.size - 1
    y_df = y.size - 1

    pooled_var = (x_ss + y_ss) / (x_df + y_df)

    se = (pooled_var / x.size + pooled_var / y.size) ** .5

    mean_diff = x.mean() - y.mean()
    t = mean_diff / se

    two_tailed = True
    alpha = .05

    prob = alpha if not two_tailed else alpha * .5

    t_critical = st.t.ppf(prob, df=(x_df + y_df))

    print('Null {}'.format('rejected' if abs(t) > abs(t_critical) else 'retained'))
    pass

def lesson_11_quiz_X():
    # mean_x = 12
    # mean_y = 8

    # n_x = 52
    # n_y = 57
    # var = 5.1

    # alpha = .05
    # two_tailed = True
    # exp_diff = 3
    # df = n_x + n_y - 2

    # prob = alpha if not two_tailed else alpha * .5
    # t_crit = st.t.ppf(prob, df=df)

    # s = var ** .5
    # mean_diff = (mean_x - mean_y) - exp_diff
    # SE = s * (1/n_x + 1/n_y) ** .5
    # # SE = (var/n_x + var/n_y) ** .5
    # t = mean_diff / SE

    # st.ttest_ind([39, 45, 48, 60], [65, 45, 32, 38])
    pass

def lesson_12_quiz_X():
    '''
    Lesson 12 | One-Way ANOVA
    '''

    df = pd.DataFrame({
        "snapzi": np.array([15, 12, 14, 11]),
        "irisa": np.array([39, 45, 48, 60]),
        "lolamoon": np.array([65, 45, 32, 38])
    })

    # Grand mean. 'df.mean().mean()' also works as all samples have same size
    df.values.flatten().mean()
    # 35.333333333333336

    # F = mst/mse = (sst/dft) / (sse/dfe)

    samples = df.values.T

    sst(samples) # 3010.666666666667
    sse(samples) # 862.0

    dft(samples) # 2
    dfe(samples) # 9

    mst(samples) # 1505.3333333333335
    mse(samples) # 95.77777777777777

    f_statistic(samples) # 15.716937354988401

    # f_critical value
    st.f.ppf(1 - .05, dfn=2, dfd=9) # 4.25649472909375

    # p_value
    1 - st.f.cdf(15.716937354988401, dfn=2, dfd=9)
    # 0.0011580762838382386

    st.f_oneway(samples[0], samples[1], samples[2])
    # (statistic=15.716937354988401, pvalue=0.0011580762838382535)

def lesson_13_quiz_X():
    df = pd.DataFrame({
        "singles": np.array([8, 7, 10, 6, 9]),
        "twins":  np.array([4, 6, 7, 4, 9]),
        "triplets": np.array([4, 4, 7, 2, 3])
    })

    samples = df.values.T

    sst(samples) # 40
    sse(samples) # 42
    dft(samples) # 2
    dfe(samples) # 12
    mst(samples) # 20
    mse(samples) # 3.5
    f_statistic(samples) # 5.714285714285714

    st.f.ppf(1-.05, dfn=2, dfd=12) # 3.8852938346523933

def lesson_14_quiz_1():
    '''
    Lesson 14 | ANOVA, Continued
    '''

    df = pd.DataFrame({
        "Food A": np.array([2, 4, 3]),
        "Food B":  np.array([6, 5, 7]),
        "Food C": np.array([8, 9, 10])
    })

    gmean = df.values.flatten().mean() # 6
    df.mean() # 3, 6, 9

    samples = df.values.T
    sst(samples) # 54.0
    sse(samples) # 6.0
    dft(samples) # 2.0
    dfe(samples) # 6.0
    mst(samples) # 27.0
    mse(samples) # 1.0
    f_statistic(samples) # 27.0
    st.f.ppf(1-.05, dfn=2, dfd=6) # 5.143252849784718

    ((samples - gmean) ** 2).flatten().sum() # 60

    q = 4.34
    tukeys_hsd = q * ((1 / 3) ** .5)

def lesson_14_quiz_22():
    '''
    Lesson 14 | ANOVA, Continued
    '''
    df = pd.DataFrame({
    "Placebo": np.array([1.5, 1.3, 1.8, 1.6, 1.3, np.nan, np.nan]),
    "Drug 1": np.array([1.6, 1.7, 1.9, 1.2, np.nan, np.nan, np.nan]),
    "Drug 2": np.array([2.0, 1.4, 1.5, 1.5, 1.8, 1.7, 1.4]),
    "Drug 3": np.array([2.9, 3.1, 2.8, 2.7, np.nan, np.nan, np.nan])
    })

    samples = df.values.T

    gmean = dropna(samples.flatten()).mean() 
    # 1.8350000000000002

    sst(samples) # 5.449428571428573
    sse(samples) # 1.0329464285714287
    dft(samples) # 3
    dfe(samples) # 16
    mst(samples) # 1.816476190476191
    mse(samples) # 0.05225446428571429
    f_statistic(samples) # 34.7621244482415

    eta2 = sst(samples) / (sst(samples) + sse(samples))
    # 0.8669841017307408

def lesson_15_quiz_17():
    '''
    Lesson 15 | Problem Set 13
    '''

    st.f.ppf(.95, dfn=3, dfd=15)
    st.f.ppf(.95, dfn=2, dfd=50)

    df = pd.DataFrame({
        "short": np.array([-8, -11, -17, -9, -10, -5, np.nan]),
        "long": np.array([12, 9, 16, 8, 15, np.nan, np.nan]),
        "normal": np.array([.5, 0, -1, 1.5, .5, -.1, 0]),
    })

    samples = df.values.T

    dropna(samples.flatten()).mean() 
    # 0.07777777777777778

    sst(samples) # 1320.171111111111
    sse(samples) # 133.48
    dft(samples) # 2
    dfe(samples) # 15
    mst(samples) # 660.0855555555555
    mse(samples) # 8.898666666666665
    f_statistic(samples) # 74.1780291679153

    st.f.ppf(.95, dfn=2, dfd=15)
    # 3.6823203436732412

    sst(samples) / (sst(samples) + sse(samples))
    # 0.908176041018554

def lesson_16_quiz_X():
    '''
    Lesson 16 | Correlation
    '''

    df = df_from_google_sheet_url(
        'https://docs.google.com/spreadsheets/d/'
        '1jMLcMJkVb_lUo5SSyI30lD9rBvtGreZZ2fWKC7dBtOk')

    df.describe()

    plt.subplot(211)
    plt.scatter(df['age'], df['party'], label='Time arrived at a party')
    plt.legend()
    plt.subplot(212)
    plt.scatter(df['age'], df['pets'], label='Pets owned')
    plt.legend()
    plt.show()

    np.corrcoef(df['age'].values, df['party'].values) # -0.16421413
    np.corrcoef(df['age'].values, df['pets'].values) # 0.37576461

    st.t.ppf(.975, df=23)

    st.pearsonr(df['age'].values, df['party'].values)
    # (-0.1642141260655671, 0.20600567570348222)
    st.pearsonr(df['age'].values, df['pets'].values)
    # (0.37576461379834125, 0.00284227543344263)

    df = df[['age', 'pets']]
    df = df.append({'age': 20, 'pets': 8}, ignore_index=True)

    st.pearsonr(df['age'].values, df['pets'].values)
    # (0.23148091642148155, 0.07025500392476225)

def lesson_17_quiz_X():
    '''
    Lesson 17 | Problem Set 14
    '''
    df = df_from_google_sheet_url(
        'https://docs.google.com/spreadsheets/d/'
        '1-M8kS83yoBG6PalNgisDDOxra-LxZbwa1Lh1K4F37pI')

    df.describe()

    written = df.iloc[:,0]
    remembered = df.iloc[:,1]

    st.pearsonr(written, remembered)
    # (0.9541449326178455, 1.21549579554e-07)

    df = df_from_google_sheet_url(
        'https://docs.google.com/spreadsheets/d/'
        '1RtrwJ7o9EwDayNl_o0UrqcnCkH9ICc1ZpieHtgZcWhg')

    df.describe()

    animal = df.iloc[:,0]
    gestation = df.iloc[:,1]
    longevity = df.iloc[:,2]

    plt.scatter(gestation, longevity, marker='v')
    plt.xlabel('Gestation')
    plt.ylabel('Longevity')
    plt.show()

    r, p = st.pearsonr(gestation, longevity)
    # (0.5893126939325756, 6.309895087810683e-05)

    r**2 # 0.3472894512300695

def lesson_18_quiz_6():
    '''
    Lesson 18 | Regression
    '''

    df = df_from_google_sheet_url(
        'https://docs.google.com/spreadsheets/d/'
        '1lzV-viazz9pPYkni6ZMsAUr0-v1jDAEDCB3VFzNYHsA')

    distance = df.iloc[:,0]
    cost = df.iloc[:,1]

    r, p = st.pearsonr(distance, cost)
    # (0.9090036493537199, 0.0006840064730744001)

    sx = distance.std() # 2315.336824548668
    sy = cost.std() # 508.1870022879811

    mean_point = distance.mean(), cost.mean()
    # (2601.1111111111113, 680.3477777777778)

    slope = r * (sy / sx) # 0.1995147465094844

    # y = slope * x + intercept
    # intercept = y - slope * x
    intercept = mean_point[1] - slope * mean_point[0]
    # 161.38775380144114

    slope * 4000 + intercept # 959.4467398393787

    linear_prediction(4000, distance, cost)
    # 959.4467398393787

    st.linregress(distance, cost)

    (500 - intercept) / slope
    # 1697.1790412618054

def lesson_19_quiz_X():
    '''
    Lesson 19 | Problem Set 15
    '''
    df = df_from_google_sheet_url(
        'https://docs.google.com/spreadsheets/d/'
        '1Xl7t5So1OEZRNSR3D_vma_CVHVBX5D64oAFDRr462Sc')

    stem = df.iloc[:,0]
    activity = df.iloc[:,1]

    plt.scatter(stem, activity)
    plt.show()

    r, p = st.pearsonr(stem, activity)
    # (0.934465030624902, 3.401445225552492e-07)

    slope = r * (activity.std() / stem.std())
    # 0.5722511760457536

    st.linregress(stem, activity)

    r ** 2 # 0.8732248934607991

    y_int = activity.mean() - slope * stem.mean()
    # -2.2487165973726846

    (70 - y_int) / slope # 126.25350479244125
    round((0 - y_int) / slope) # 4

    2.35 + .05 * 10 # 2.85
    2.35 + .05 * 5 # 2.6
    2.35 + .05 * 0 # 2.35
    2.35 + .05 * 20 # 3.35

    se = 3.5
    #t_crit = 1.984
    #margin_of_error = se * t_crit # 6.759060545297078

    (2.85- se, 2.85 + se) # (6.5, 13.5)
    (2.6 - se, 2.6 + se) # (1.5, 8.5)
    (2.35 - se, 2.35 + se) # (-3.5, 3.5)
    (3.35 - se, 3.35 + se) # (16.5, 23.5)

def lesson_20_quiz_16():
    '''
    Lesson 20 | X^2 Tests
    '''
    n = 100  
    df = pd.DataFrame({
        'expected': [33, 67],
        'observed': [41, 59],
    })
    
    df.set_index(
        pd.Index(['successful', 'unsuccessful']), 
        inplace=True
    )

    (((df['observed'] - df['expected']) ** 2) / df['expected']).sum()
    # 2.8946178199909545

    chi2, p = st.chisquare(df['observed'], df['expected'])
    # 2.8946178199909545, 0.08887585044058065

def lesson_21_quiz_X():
    '''
    Lesson 20 | Problem Set 16
    '''
    st.chisquare([8, 4, 1, 8, 3, 0])
    # Power_divergenceResult(statistic=14.5, pvalue=0.012726685122400083)
    st.chi2.ppf(.95, df=5)
    # 11.070497693516351

    st.chi2.ppf(.95, df=1)
    # 3.841458820694124

    st.chi2_contingency([[299, 186], [280, 526]], correction=False)
    # (88.6487963871521, 4.7151344226599645e-21, 1,  ...)

