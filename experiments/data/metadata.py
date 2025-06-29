import dotenv, os

dotenv.load_dotenv()



droot = os.environ.get("DATA_PATH") 




feat_dict ={'law_school': {'cat': ['fam_inc', 'tier'	],
                          'bool':['male', 'race', 'fulltime'],
                          'num' :['decile1b','decile3','lsat','ugpa','zfygpa','zgpa'],
                          'sens_attrs' :['male', 'race'],
                          'sens_attrs_map':{'male':{1:'Male',0:'Female' }, 'race':{1:'Caucasian', 0:'African-American'}},
                          'target':'pass_bar',
                          'n_subgroups' :4,
                          'favorable_outcome' : 1,
                          'target_map': {1: 'Pass', 0: 'Fail'}
                          },

            'credit_card':{'cat': [],
                          'bool':['SEX',	'MARRIAGE'	],
                          'num' :['LIMIT_BAL', 	'AGE', 	'PAY_0', 	'PAY_2'	, 'PAY_3', 	'PAY_4', 	'PAY_5', 	'PAY_6'	, 
                                  'BILL_AMT1', 	'BILL_AMT2', 	'BILL_AMT3', 	'BILL_AMT4', 	'BILL_AMT5'	, 'BILL_AMT6', 
                                  'PAY_AMT1', 	'PAY_AMT2', 	'PAY_AMT3', 	'PAY_AMT4', 	'PAY_AMT5', 	'PAY_AMT6'],
                          'sens_attrs' :['SEX',	'MARRIAGE'	],
                          'sens_attrs_map':{'SEX':{1:'Male',0:'Female' }, 'MARRIAGE':{1:'Married', 0:'Single'}},
                          'target':'default payment next month',
                          'n_subgroups' :4,
                          'favorable_outcome' : 1,
                          'target_map': {1: 'Default', 0: 'No Default'}
                          },

            'adult':{'cat': [],
                          'bool':['sex','race'	, 'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Private',
                                'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
                                'workclass_State-gov', 'workclass_Without-pay', 'education_10th',
                                'education_11th', 'education_12th', 'education_1st-4th',
                                'education_5th-6th', 'education_7th-8th', 'education_9th',
                                'education_Assoc-acdm', 'education_Assoc-voc', 'education_Bachelors',
                                'education_Doctorate', 'education_HS-grad', 'education_Masters',
                                'education_Preschool', 'education_Prof-school',
                                'education_Some-college', 'marital-status_Divorced',
                                'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse',
                                'marital-status_Married-spouse-absent', 'marital-status_Never-married',
                                'marital-status_Separated', 'marital-status_Widowed',
                                'occupation_Adm-clerical', 'occupation_Armed-Forces',
                                'occupation_Craft-repair', 'occupation_Exec-managerial',
                                'occupation_Farming-fishing', 'occupation_Handlers-cleaners',
                                'occupation_Machine-op-inspct', 'occupation_Other-service',
                                'occupation_Priv-house-serv', 'occupation_Prof-specialty',
                                'occupation_Protective-serv', 'occupation_Sales',
                                'occupation_Tech-support', 'occupation_Transport-moving',
                                'relationship_Husband', 'relationship_Not-in-family',
                                'relationship_Other-relative', 'relationship_Own-child',
                                'relationship_Unmarried', 'relationship_Wife',
                                'native-country_Cambodia', 'native-country_Canada',
                                'native-country_China', 'native-country_Columbia',
                                'native-country_Cuba', 'native-country_Dominican-Republic',
                                'native-country_Ecuador', 'native-country_El-Salvador',
                                'native-country_England', 'native-country_France',
                                'native-country_Germany', 'native-country_Greece',
                                'native-country_Guatemala', 'native-country_Haiti',
                                'native-country_Holand-Netherlands', 'native-country_Honduras',
                                'native-country_Hong', 'native-country_Hungary', 'native-country_India',
                                'native-country_Iran', 'native-country_Ireland', 'native-country_Italy',
                                'native-country_Jamaica', 'native-country_Japan', 'native-country_Laos',
                                'native-country_Mexico', 'native-country_Nicaragua',
                                'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru',
                                'native-country_Philippines', 'native-country_Poland',
                                'native-country_Portugal', 'native-country_Puerto-Rico',
                                'native-country_Scotland', 'native-country_South',
                                'native-country_Taiwan', 'native-country_Thailand',
                                'native-country_Trinadad&Tobago', 'native-country_United-States',
                                'native-country_Vietnam', 'native-country_Yugoslavia'],
                          'num' :['age','education-num', 'capital-gain','capital-loss', 'hours-per-week' ],
                          'sens_attrs' :['sex','race'	],
                          'sens_attrs_map':{'sex':{1:'Male',0:'Female' }, 'race':{1:'Caucasian', 0:'African-American'}},
                          'target':'salary-class',
                          'n_subgroups' :4,
                          'favorable_outcome' : 1,
                          'target_map': {1: '>50K', 0: '<=50K'}
                          },


            'compas':{'cat': [],
                          'bool':['sex','race', 'c_charge_degree'],
                          'num' :['age', 'priors_count', 'length_of_stay', 'juv_fel_count', 'juv_misd_count', 'juv_other_count'],
                          'sens_attrs' :['sex','race'],
                          'sens_attrs_map':{'sex':{1:'Male',0:'Female' }, 'race':{1:'Caucasian', 0:'African-American'}},
                          'target':'two_year_recid',
                          'n_subgroups' :4,
                          'favorable_outcome' : 0,
                          'target_map': {1: 'Recidivism', 0: 'No Recidivism'}
                          },
            }


adult_features = [
    'age',
    'workclass',
    'fnlwgt',
    'education', 
    'education-num', 
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex', 
    'capital-gain', 
    'capital-loss', 
    'hours-per-week', 
    'native-country',
    'salary-class'
]

compas_features = ['age', 'c_charge_degree', 'race',  'score_text',
                'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score',
                'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out', 
                'juv_fel_count', 'juv_misd_count', 'juv_other_count']


law_features =  ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'fulltime', 'fam_inc', 'male', 'tier', 'race', 'pass_bar']