from airflow import DAG
from datetime import datetime
from datetime import date
from airflow.operators.python_operator import PythonOperator
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing as preprocessing


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'start_date':datetime(2020,12,25)
}

dag = DAG(
    dag_id='heisenburgs_dag',
    default_args=default_args,
    description='project',
    schedule_interval='@once',
    
)

def extract(**kwargs):
    happiness_2015_df = pd.read_csv("/home/beltagy/airflow/dags/data/Happiness_Dataset/2015.csv")
    happiness_2016_df = pd.read_csv("/home/beltagy/airflow/dags/data/Happiness_Dataset/2016.csv")
    happiness_2017_df = pd.read_csv("/home/beltagy/airflow/dags/data/Happiness_Dataset/2017.csv")
    happiness_2018_df = pd.read_csv("/home/beltagy/airflow/dags/data/Happiness_Dataset/2018.csv")
    happiness_2019_df = pd.read_csv("/home/beltagy/airflow/dags/data/Happiness_Dataset/2019.csv")
    df_countries=pd.read_csv('/home/beltagy/airflow/dags/data/250 Country Data.csv')
    df_life_expectancy=pd.read_csv("/home/beltagy/airflow/dags/data/Life Expectancy Data.csv")
    df_list=[happiness_2015_df,happiness_2016_df,happiness_2017_df,happiness_2018_df,happiness_2019_df,df_countries,df_life_expectancy]
    return (df_list)

def transform (**context):
    df_list=context['task_instance'].xcom_pull(task_ids='extract')
    happiness_2015_df,happiness_2016_df,happiness_2017_df,happiness_2018_df,happiness_2019_df,df_countries,df_life_expectancy=df_list
    happiness_2015_df["Year"]=2015
    happiness_2016_df["Year"]=2016
    happiness_2017_df["Year"]=2017
    happiness_2018_df["Year"]=2018
    happiness_2019_df["Year"]=2019
    happiness_2015_df = happiness_2015_df.drop(["Region", "Standard Error", 'Dystopia Residual'], axis=1)
    happiness_2015_df= happiness_2015_df.rename(columns={"Country":"Country or region",
                                          "Happiness Score": "Score", 
                                          "Happiness Rank":"Overall rank", 
                                          "Economy (GDP per Capita)":"GDP per capita",
                                          "Health (Life Expectancy)" : "Healthy life expectancy",
                                          "Trust (Government Corruption)": "Perceptions of corruption",
                                          "Family":"Social support",
                                          "Freedom": "Freedom to make life choices"})
    happiness_2016_df = happiness_2016_df.drop(["Region", 'Dystopia Residual', 'Lower Confidence Interval', 'Upper Confidence Interval'], axis=1)
    happiness_2016_df = happiness_2016_df.rename(columns={"Country":"Country or region",
                                          "Happiness Score": "Score", 
                                          "Happiness Rank":"Overall rank", 
                                          "Economy (GDP per Capita)":"GDP per capita",
                                          "Health (Life Expectancy)" : "Healthy life expectancy",
                                          "Trust (Government Corruption)": "Perceptions of corruption",
                                          "Family":"Social support",
                                          "Freedom": "Freedom to make life choices"})
    happiness_2017_df=happiness_2017_df.drop(['Whisker.high', 'Whisker.low', 'Dystopia.Residual'], axis=1)
    happiness_2017_df = happiness_2017_df.rename(columns={"Country":"Country or region",
                                          "Happiness.Score": "Score", 
                                          "Happiness.Rank":"Overall rank", 
                                          "Economy..GDP.per.Capita.":"GDP per capita",
                                          "Health..Life.Expectancy." : "Healthy life expectancy",
                                          "Trust..Government.Corruption.": "Perceptions of corruption",
                                          "Family":"Social support",
                                          "Freedom": "Freedom to make life choices"})
    happiness_df = pd.concat([happiness_2015_df,happiness_2016_df,happiness_2017_df,happiness_2018_df,happiness_2019_df])
    happiness_df = happiness_df.replace('Hong Kong S.A.R., China','Hong Kong')
    happiness_df = happiness_df.replace('Somaliland region','Somaliland Region')
    happiness_df = happiness_df.replace('Taiwan Province of China','Taiwan')
    happiness_df = happiness_df.replace('Trinidad & Tobago','Trinidad and Tobago')

    # end of part 1 happiness data.
    df_countries.drop(['Unnamed: 0'],axis=1,inplace=True)
    df_countries['Real Growth Rating(%)']= pd.Series(df_countries['Real Growth Rating(%)'].str.extract(r"([-+]?\d*\.*\d+|\d+)",expand=False).values.astype(float))*.01
    df_countries['Literacy Rate(%)']= pd.Series(df_countries['Literacy Rate(%)'].str.extract(r"([-+]?\d*\.*\d+|\d+)",expand=False).values.astype(float))*.01
    df_countries['Inflation(%)']= pd.Series(df_countries['Inflation(%)'].str.extract(r"([-+]?\d*\.*\d+|\d+)",expand=False).values.astype(float))*.01
    df_countries['Unemployement(%)']=pd.Series( df_countries['Unemployement(%)'].str.extract(r"([-+]?\d*\.*\d+|\d+)",expand=False).values.astype(float))*.01
    df_countries.drop(df_countries[df_countries.subregion.isna()].index,inplace=True)

    # invistigating the coulmns with missing null values and their percentages
    nans = pd.DataFrame(data=[], index=None, 
                          columns=['feature_name','missing_values','percentage_of_total'])
    nans['feature_name'] = df_countries.columns[df_countries.isna().sum()>0]
    nans['missing_values'] = np.array(df_countries[nans.iloc[:,0]].isna().sum())
    nans['percentage_of_total'] = np.round(nans['missing_values'] / df_countries.shape[0] * 100)
    nans['var_type']= [df_countries[c].dtype for c in nans['feature_name']]

    # filling in the missing values

    nan_cols = nans['feature_name']
    for col in nan_cols:
        df_countries[col] = df_countries.groupby("region").transform(lambda x: x.fillna(x.mean()))[col]

    # exploring and cleaning expectancy data

    df_life_expectancy = df_life_expectancy.drop(' thinness 5-9 years', axis=1)
    #columns with null values
    nans = pd.DataFrame(data=[], index=None, 
                            columns=['feature_name','missing_values','percentage_of_total'])
    nans['feature_name'] = df_life_expectancy.columns[df_life_expectancy.isna().sum()>0]
    nans['missing_values'] = np.array(df_life_expectancy[nans.iloc[:,0]].isna().sum())
    nans['percentage_of_total'] = np.round(nans['missing_values'] / df_life_expectancy.shape[0] * 100)
    nans['var_type']= [df_life_expectancy[c].dtype for c in nans['feature_name']]

    nan_cols = nans['feature_name'][nans['missing_values']<=50]
    for col in nan_cols:
        mean_ = df_life_expectancy[col].mean()
        df_life_expectancy.loc[df_life_expectancy[col].isna()==True,col]=mean_

    nans = pd.DataFrame(data=[], index=None, 
                          columns=['feature_name','missing_values','percentage_of_total'])
    nans['feature_name'] = df_life_expectancy.columns[df_life_expectancy.isna().sum()>0]
    nans['missing_values'] = np.array(df_life_expectancy[nans.iloc[:,0]].isna().sum())
    nans['percentage_of_total'] = np.round(nans['missing_values'] / df_life_expectancy.shape[0] * 100)
    nans['var_type']= [df_life_expectancy[c].dtype for c in nans['feature_name']]

    # imputing the missing values.
    impute(df_life_expectancy, 'GDP', 'Total expenditure')
    impute(df_life_expectancy, 'Total expenditure', 'GDP')
    df_life_expectancy.loc[df_life_expectancy['GDP'].isna() == True,'GDP'] =df_life_expectancy['GDP'].mean()
    df_life_expectancy.loc[df_life_expectancy['Total expenditure'].isna()==True,'Total expenditure']=df_life_expectancy['Total expenditure'].mean()

    impute(df_life_expectancy, 'Alcohol', 'Schooling')
    impute(df_life_expectancy, 'Schooling', 'Alcohol')
    df_life_expectancy.loc[df_life_expectancy['Alcohol'].isna() == True,'Alcohol'] =df_life_expectancy['Alcohol'].mean()
    df_life_expectancy.loc[df_life_expectancy['Schooling'].isna()==True,'Schooling']=df_life_expectancy['Schooling'].mean()

    impute(df_life_expectancy, 'Hepatitis B', 'Diphtheria ')
    df_life_expectancy.loc[df_life_expectancy['Hepatitis B'].isna() == True,'Hepatitis B'] =df_life_expectancy['Hepatitis B'].mean()

    impute(df_life_expectancy, 'Population', 'infant deaths')
    df_life_expectancy.loc[df_life_expectancy['Population'].isna() == True,'Population'] =df_life_expectancy['Population'].mean()

    impute(df_life_expectancy, 'Income composition of resources', 'Schooling')
    df_life_expectancy.loc[df_life_expectancy['Income composition of resources'].isna() == True,'Income composition of resources'] =df_life_expectancy['Income composition of resources'].mean()

    # removing the outliers 

    cols=df_life_expectancy.columns.tolist()
    cols.remove('Country')
    cols.remove('Year')
    cols.remove('Status')
    for col in cols:
        df_life_expectancy=outlier_replace(col,df_life_expectancy)

    
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['United States'],'United States of America')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['United Kingdom'],'United Kingdom of Great Britain and Northern Ireland')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Venezuela'],'Venezuela (Bolivarian Republic of)')
    'Bolivia (Plurinational State of)'
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Bolivia'],'Bolivia (Plurinational State of)')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Moldova'],'Moldova (Republic of)')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Russia'],'Russian Federation')

    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Kosovo'],'Republic of Kosovo')
    # here we change in df_coutries dataset
    df_countries['name'] = df_countries['name'].replace(['Viet Nam'],'Vietnam')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Macedonia'],'Macedonia (the former Yugoslav Republic of)')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Laos'],"Lao People's Democratic Republic")
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Palestinian Territories'],'Palestine, State of')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Iran'],'Iran (Islamic Republic of)')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Tanzania'],'Tanzania, United Republic of')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Ivory Coast'],'Côte d\'Ivoire')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Syria'],'Syrian Arab Republic')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['South Korea'],"Korea (Democratic People's Republic of)")
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Congo (Kinshasa)'],'Congo (Democratic Republic of the)')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['Congo (Brazzaville)'],'Congo')
    happiness_df['Country or region'] = happiness_df['Country or region'].replace(['North Macedonia'],'Macedonia (the former Yugoslav Republic of)')
    df_countries_happiness=pd.merge(df_countries,happiness_df,left_on="name",right_on='Country or region')
    df_countries_happiness=df_countries_happiness.drop(['Country or region'],axis=1).rename(columns={"name": "country"})

    # merging the life-expectancy data with the country data

    df_life_expectancy['Country'] = df_life_expectancy['Country'].replace(['Czechia'],'Czech Republic')
    df_life_expectancy['Country'] = df_life_expectancy['Country'].replace(['Democratic People\'s Republic of Korea'],"Korea (Democratic People's Republic of)")
    df_life_expectancy['Country'] = df_life_expectancy['Country'].replace(['Republic of Korea'],'Korea (Republic of)')
    df_life_expectancy['Country'] = df_life_expectancy['Country'].replace(['Republic of Korea'],'Korea (Republic of)')
    df_life_expectancy['Country'] = df_life_expectancy['Country'].replace(['The former Yugoslav republic of Macedonia'],'Macedonia (the former Yugoslav Republic of)')
    df_life_expectancy['Country'] = df_life_expectancy['Country'].replace(['United Republic of Tanzania'],'Tanzania, United Republic of')
    df_life_expectancy['Country'] = df_life_expectancy['Country'].replace(['Viet Nam'],'Vietnam')
    df_life_expectancy['Country'] = df_life_expectancy['Country'].replace(['Democratic Republic of the Congo'],'Congo (Democratic Republic of the)')

    df_countries_expectancy=pd.merge(df_countries,df_life_expectancy,left_on="name",right_on='Country')
    df_countries_expectancy=df_countries_expectancy.drop(['Country'],axis=1).rename(columns={"name": "country"})

    # first feature engineering
    df_countries_expectancy['Status']=df_countries_expectancy['Status'].apply(lambda x: 0 if x=="Developing" else 1).astype(int)
    #second feature engineering
    df_countries_expectancy['region']=preprocessing.LabelEncoder().fit_transform(df_countries_expectancy['region'])
    df_countries_expectancy['subregion']=preprocessing.LabelEncoder().fit_transform(df_countries_expectancy['subregion'])
    df_countries_happiness['region']=preprocessing.LabelEncoder().fit_transform(df_countries_happiness['region'])
    df_countries_happiness['subregion']=preprocessing.LabelEncoder().fit_transform(df_countries_happiness['subregion'])
    # third fearure engineering
    df_countries_expectancy['population_desity']=df_countries_expectancy['population']/df_countries_expectancy['area']
    df_countries_happiness['population_desity']=df_countries_happiness['population']/df_countries_happiness['area']
    # fourth feature engineering
    df_countries_happiness['population']=MinMaxScaler().fit_transform(df_countries_happiness[["population"]]) 
    df_countries_happiness['area']=MinMaxScaler().fit_transform(df_countries_happiness[["area"]]) 
    df_countries_expectancy['population']=MinMaxScaler().fit_transform(df_countries_expectancy[["population"]]) 
    df_countries_expectancy['area']=MinMaxScaler().fit_transform(df_countries_expectancy[["area"]]) 

    merged_list=[df_countries_happiness,df_countries_expectancy]
    return(merged_list)

























def impute(df, to_impute, reference):
    index=df[to_impute][(df[to_impute].isna()==True)&
                    (df[reference].isna()==False)].keys()
    #df['Total expenditure'][index]
    var_min = df[reference].min()
    var_max = df[reference].max()
    range_filler =  var_max - var_min
    step = range_filler / 10
    one = df[to_impute][df[reference] < (var_min+step)].mean()
    two = df[to_impute][(df[reference] > (var_min+step))&
              (df[reference] < (var_min+step*2))].mean()
    three = df[to_impute][(df[reference] > (var_min+step*2))&
              (df[reference] < (var_min+step*3))].mean()
    four = df[to_impute][(df[reference] > (var_min+step*3))&
              (df[reference] < (var_min+step*4))].mean()
    five = df[to_impute][(df[reference] > (var_min+step*4))&
              (df[reference] < (var_min+step*5))].mean()
    six = df[to_impute][(df[reference] > (var_min+step*5))&
              (df[reference] < (var_min+step*6))].mean()
    seven = df[to_impute][(df[reference] > (var_min+step*6))&
              (df[reference] < (var_min+step*7))].mean()
    eight = df[to_impute][(df[reference] > (var_min+step*7))&
              (df[reference] < (var_min+step*8))].mean()
    nine = df[to_impute][(df[reference] > (var_min+step*8))&
              (df[reference] < (var_min+step*9))].mean()
    ten = df[to_impute][df[reference] > (var_max-step)].mean()
    
    for i in index:
        if df[reference][i] < (var_min+step):
            df[to_impute][i]=one
        elif df[reference][i] < (var_min+step*2):
                df[to_impute][i]=two
                continue
        elif df[reference][i] < (var_min+step*3):
                df[to_impute][i]=three
                continue
        elif df[reference][i] < (var_min+step*4):
                df[to_impute][i]=four
                continue
        elif df[reference][i] < (var_min+step*5):
                df[to_impute][i]=five
                continue
        elif df[reference][i] < (var_min+step*6):
                df[to_impute][i]=six
                continue
        elif df[reference][i] < (var_min+step*7):
                df[to_impute][i]=seven
                continue
        elif df[reference][i] < (var_min+step*8):
                df[to_impute][i]=eight
                continue
        elif df[reference][i] < (var_min+step**9):
                df[to_impute][i]=nine
                continue
        else:
            df[to_impute][i]=ten


def outlier_replace(col,df_life_expectancy):
    countries=df_life_expectancy['Country'].unique()
    groups=df_life_expectancy.groupby('Country')
    count=0
    for i in countries:
        for j in groups.get_group(i)[col]:
            threshold = 3
            mean = np.mean(groups.get_group(i)[col])
            std = np.std(groups.get_group(i)[col])
            if std != 0:                     
                z_score = (j - mean) / std
                if np.abs(z_score) > threshold:
                    df_life_expectancy.loc[df_life_expectancy['Country']==i,col] = df_life_expectancy[col][df_life_expectancy['Country'] == i].mean()
                    count+=1
    return df_life_expectancy


def load_data(**context):
        df_list=context['task_instance'].xcom_pull(task_ids='transform')
        df_countries_happiness,df_countries_expectancy=df_list
        df_countries_happiness.to_csv('/home/beltagy/airflow/dags/data/countries_happiness')
        df_countries_expectancy.to_csv('/home/beltagy/airflow/dags/data/countries_happiness')

t1=PythonOperator(
    task_id='extract',
    provide_context=True,
    python_callable=extract,
    dag=dag
)

t2=PythonOperator(
    task_id='transform',
    provide_context=True,
    python_callable = transform,
    dag=dag
)

t3=PythonOperator(
    task_id='load_data',
    provide_context=True,
    python_callable = load_data,
    dag=dag
)


t1>>t2
t2>>t3
