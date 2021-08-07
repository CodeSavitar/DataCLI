import datetime
import time
import numpy as np
from numpy import nan
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import click
from pyfiglet import Figlet
from click_help_colors import HelpColorsGroup, HelpColorsCommand

data = pd.read_excel('Data.xlsx')
df = data.copy()

@click.group(
    cls=HelpColorsGroup, help_headers_color="yellow", help_options_color="cyan"
)
@click.version_option("0.1.7", prog_name="cli")
def main():
    """DataCLI : A simple CLI where you can do your preprocessing"""
    pass


@main.group()
def process():
    """Types of pre processing methods:- Display, Description, Null-Values, Categorical-Data"""
    pass


@process.command("Display")
def disp_data():
    click.echo("Displaying Dataset")
    click.echo(click.style("Accessed time::",fg="red") + "{}".format(datetime.datetime.now()))
    click.echo(df.head(30))


@process.command("Description")
def desc_data():
    click.echo("Dataset Description")
    click.echo(click.style("Accessed time::",fg="red") + "{}".format(datetime.datetime.now()))
    click.echo(df.describe())    


@process.command("Column-Drop")
@click.argument("column")
def drop(column):
    click.echo("Dropping the preferred column")
    click.echo(click.style("Accessed time::",fg="red") + "{}".format(datetime.datetime.now()))
    df.drop(column, inplace=True, axis=1)
    click.echo(df.head(30))


@process.command("Null-Values")
@click.argument("column")
def null(column):
    click.echo("Handling the Null Values")
    click.echo(click.style("Accessed time::",fg="red") + "{}".format(datetime.datetime.now()))

    tot = df[column].isnull().sum()
    if tot > 0:
        df[[column]] = df[[column]].replace(0, nan)
        df.fillna(df.mean(),inplace=True)
        click.echo(df.head(30))
    else:
        click.echo("There are no null values in this column")    


@process.command("Categorical-Data")
@click.argument("column")
def cat(column):
    click.echo("Encoding Categorical Data")
    click.echo(click.style("Accessed time::",fg="red") + "{}".format(datetime.datetime.now()))

    #click.echo("Encoding Independent Data")
    #x = df[column]
    '''ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
    x = np.array(ct.fit_transform(x))'''
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    click.echo(df.head(30))


@process.command("Data-Preprocess")
def preprocess():
    click.echo("Preparing the dataset")
    click.echo(click.style("Accessed time::",fg="red") + "{}".format(datetime.datetime.now()))

    count1 = 0
    count2 = 0

    for i in df:
        if df[i].isnull().sum() > 0:
            count1 = count1 + df[i].isnull().sum()
            df[i] = df[i].replace(0, nan)
            df.fillna(df.mean(),inplace=True)

    for i in df:
        if df[i].dtypes == 'object':
            count2 = count2 + 1
            le = LabelEncoder()
            df[i] = le.fit_transform(df[i])

    z = np.abs(stats.zscore(df))
    data = df[(z < 5).all(axis=1)]

    shape1 = df.shape
    shape2 = data.shape        

    file_name = "Processed Data.csv"
    with click.progressbar(range(5),label='Downloading Dataset:') as bar:
        for i in bar:
            data.to_csv(file_name, index=False)        

    click.echo("Dataset downloaded")
    #click.echo(shape1,shape2)
    click.echo("Null Values encountered : {}".format(count1))
    click.echo("Categorical Columns encountered : {}".format(count2))
    click.echo("Outliers encountered : {}".format(shape1[0] - shape2[0]))  


@main.command()
def info():
    f = Figlet(font='standard')
    click.echo(f.renderText('ML Data Preprocessing'))
    click.secho("CLI : A simple CLI where you can do your preprocessing",fg='cyan')
    click.echo("Author: Adhith Sankar")    


if __name__ == "__main__":
    main()

