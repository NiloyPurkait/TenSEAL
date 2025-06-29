import os, sys, dotenv
import numpy as np

dotenv.load_dotenv()
root_path = os.environ.get("SCRIPT_PATH")

sys.path.append(root_path)
import pandas as pd
import numpy as np


def get_adult():
    '''
       Source: Extraction was done by Barry Becker from the 1994 Census database.
       URL: https://archive.ics.uci.edu/dataset/2/adult
    
    '''
    sens_attrs = {
                  'sex': {'Male': 1, 'Female': 0},
                  'race': {'Black': 0, '*': 1}
                 }
    return (adult_preproc()
                        .astype({'age': float, 'education-num': float,
                                 'salary-class': int, 'capital-gain': float, 
                                 'capital-loss': float, 'hours-per-week': int})
                        .pipe(encode_sens_cols, sens_attrs)
                        .pipe(lambda d: pd.get_dummies(d, columns=list(d.select_dtypes('O').columns), dtype=np.float32))
        )



def get_compas():
    '''
       Source: ProPublica's analysis of COMPAS recidivism scores
       URL: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
    '''
    from data.metadata import droot
    compas_path = f'{droot}/compas/compas-scores-two-years.csv'
    target = 'two_year_recid'
    df = (pd.read_csv(compas_path, delimiter=',')
          .pipe(compas_preproc))
    df.index.name='ID'
    t =  df[target]
    df = df.drop(target, axis=1)
    df.insert(df.shape[1] ,target,t)
    df.sex = df.sex.replace({'Male':1, 'Female':0})
    df.race = df.race.replace({'African-American':0, 'Caucasian':1})
    df.c_charge_degree = df.c_charge_degree.replace({'M':0, 'F':1})
    return df


def get_law_school():
    '''
        Source: Collected for "LSAC National Longitudinal Bar Passage Study" by Linda Wightman in 1998.
        URL: https://www.kaggle.com/datasets/danofer/law-school-admissions-bar-passage
    '''
    from data.metadata import droot, law_features
    law_path = f'{droot}/law/bar_pass_prediction.csv'
    df =  pd.read_csv(law_path, index_col=0)[law_features].dropna(axis=0).astype({'male':'int'})
    df.race = np.where(df.race.values==7,1,0)
    return df.reset_index(drop=True)



def get_credit_card():
    ''' 
        Source: I-Cheng Yeh, "The comparisons of data mining techniques for the predictive
                              accuracy of probability of default of credit card clients",
                              Expert Systems with Applications, 2009.

        URL : https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients'''
    from data.metadata import droot
    
    df= pd.read_excel(f'{droot}/credit card clients/default of credit card clients.xlsx', engine='openpyxl')
    df.columns = df.loc[0, :]
    df = df.drop(0, axis=0)

    return (df.replace({'SEX':{1:1, 2:0}, 
                    
                'EDUCATION': {1:'grad. school',
                                2:'university',
                                3:'high school',
                                4:'other'},

                'MARRIAGE':{1:1,#'married',
                            2:0,#'single',
                            3:'other'}})
            .pipe(lambda df: df[df.MARRIAGE!='other'])

            .astype({**{'SEX':int, 'LIMIT_BAL':float, 'MARRIAGE':int,
                        'AGE':float,'default payment next month':int },
                    **{i: float for i in df.filter(regex='BILL').columns},
                    **{i: float for i in df.filter(regex='PAY').columns}})
            .drop('ID', axis=1)
            .pipe(lambda df : pd.get_dummies(df, columns=df.select_dtypes('O').columns))
            
    )




##########################################
# Helper functions to preprocess datasets#
##########################################


def compas_preproc(df):
    """The custom pre-processing function is adapted from
        https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb

    """
    from data.metadata import compas_features
    df = df[compas_features]
    df = df.assign(juv_count= df.apply(lambda x:  x.juv_fel_count  \
                                                + x.juv_misd_count \
                                                + x.juv_other_count,
                                       axis=1))
    # Indices of data samples to keep
    ix = df['days_b_screening_arrest'] <= 30
    ix = (df['days_b_screening_arrest'] >= -30) & ix
    ix = (df['is_recid'] != -1) & ix
    ix = (df['c_charge_degree'] != "O") & ix
    ix = (df['score_text'] != 'N/A') & ix
    df = df.loc[ix,:]
    df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-
                            pd.to_datetime(df['c_jail_in'])).apply(
                                                    lambda x: int(round(x.days/30, 0)))
    # Restrict races to African-American and Caucasian
    df = df[(df.race=='African-American') | (df.race=='Caucasian') ]
    # Restrict the features to use
    df = df.loc[:, ['sex','race','age','c_charge_degree','priors_count',
                    'two_year_recid','length_of_stay', 'juv_fel_count', 
                    'juv_misd_count', 'juv_other_count']]
    df.index.name = 'ID'
    return df




def adult_preproc():
    from data.metadata import droot, adult_features
    adult_data_dir = f'{droot}/adult/adult.data'
    adult_test_dir = f'{droot}/adult/adult.test'

    # Load datasets
    adult_data = np.loadtxt(adult_data_dir, dtype=object, delimiter=',')
    adult_test = np.loadtxt(adult_test_dir, dtype=object, delimiter=',')

    # Combine datasets
    adult_combined = adult_data

    # Convert to DataFrame
    adult_combined = pd.DataFrame(adult_combined, columns=adult_features).drop(['fnlwgt'], axis=1)
    adult_combined = adult_combined.apply(lambda x: x.str.strip())
    adult_combined['salary-class'] = np.where(adult_combined['salary-class'] == '>50K', 1, 0)

    adult_combined = adult_combined.replace('?', np.nan).dropna()
    adult_combined.index.name = 'ID'

    return adult_combined




def encode_sens_cols(df, sens_cols):
    for sens_attr, value_codes in sens_cols.items():
        wildcard_checker = [v for v, c in value_codes.items()]
        if '*' in wildcard_checker:
            for v, c in value_codes.items():
                if v!='*':
                    df[sens_attr] = np.where(df[sens_attr].values == v, value_codes[v], int(not value_codes[v])).astype(int)
        else:
            df[sens_attr] = df[sens_attr].replace(value_codes).astype(int)
    return df





