import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import argparse

# example use:
# python investigate_emdb_data.py -i emdb_data.txt -o '.' -p True -b True

def read_csv(csv_file)->pd.DataFrame:
    df = pd.read_csv(csv_file, dtype=object)
    return df

def print_df(df, all_entries=False):
    if all_entries:
        pd.set_option('display.max_rows', None)
    print(df)
    pd.set_option('display.max_rows', 20)
    return

def drop_df_entries(df, col, drop_on=',')->pd.DataFrame:
    print('dropped entries are:')
    print_df(df[df[col].str.contains(drop_on)])
    df_temp = df[~df[col].str.contains(drop_on)]
    return df_temp

def create_boxplots(df, outpath, label='_boxplot', file_ext=['.pdf','.png'], exclude_cols=['emdb_id','empiar_id']):
    
    #grab df columns with numeric data only
    print('Data Types:\n{}'.format(df.dtypes))

    # grab df columns
    df_columns = df.columns.values.tolist()

    for exclude in exclude_cols:
        if exclude in df_columns:
            df=df.drop(exclude, axis=1)
    # print_df(df, all_entries=True)
    
    # grab df columns
    df_columns = df.columns.values.tolist()
    for col in df_columns:
        print('Col: {}'.format(col))
        # drop rows containing multiple entries in this field (identified with a comma separation)
        boxplot_df = drop_df_entries(df, col)
        # convert remaining cols to dtype float or int
        boxplot_df[col] = pd.to_numeric(boxplot_df[col])
        print('avg of entries in {}: {}'.format(col, boxplot_df[col].mean()))
        axes, dict= boxplot_df.boxplot(column=col, fontsize=18, grid=True, figsize=(12,12), return_type='both', color='r')
        # now print it out
        plt.tight_layout()
        for ext in file_ext:
            outfile=os.path.join(outpath, col+label+ext)
        plt.savefig(outfile, dpi=600)
        print('Saved: {}'.format(outfile))
        plt.clf()
    return

def create_hists(df, outpath, label='_hist', file_ext=['.pdf', '.png'], exclude_cols=['emdb_id','empiar_id']):
        #grab df columns with numeric data only
    print('Data Types:\n{}'.format(df.dtypes))
    # df = df.select_dtypes('number')
    # get rid of embd_id column using if default args
    # grab df columns
    df_columns = df.columns.values.tolist()

    for exclude in exclude_cols:
        if exclude in df_columns:
            df=df.drop(exclude, axis=1)
    # print_df(df, all_entries=True)
    
    # grab df columns
    df_columns = df.columns.values.tolist()
    for col in df_columns:
        print('Col: {}'.format(col))
        # drop rows containing multiple entries in this field (identified with a comma separation)
        boxplot_df = drop_df_entries(df, col)
        # print_df(boxplot_df, all_entries=True)
        # convert remaining cols to dtype float or int
        boxplot_df[col] = pd.to_numeric(boxplot_df[col])
        print('avg of entries in {}: {}'.format(col, boxplot_df[col].mean()))
        axes= boxplot_df.hist(column=col, xlabelsize=18, ylabelsize=18, grid=True, figsize=(12,12), color='r', bins=100)

        plt.tight_layout()

        for ext in file_ext:
            outfile=os.path.join(outpath, col+label+ext)
        plt.savefig(outfile, dpi=600)
        print('Saved: {}'.format(outfile))
        plt.clf()
    return

def run_main(emdb_file):
    if args.input_csv is not None:
        emdb_df=read_csv(args.input_csv)
        if args.print_csv is not None:
            print_df(emdb_df)
        if args.create_boxplots is not None:
            create_boxplots(emdb_df, args.output_path)
        if args.create_hists is not None:
            create_hists(emdb_df, args.output_path)
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_csv", help="The csv file full of emdb data taken from https://www.ebi.ac.uk/emdb/",\
    type=str, default="/mnt/parakeet_storage/trajectories/trajic/parakeet_scripts")
    parser.add_argument("-o", "--output_path", help="Path to directory to save plots into", type=str, default="")
    parser.add_argument("-p", "--print_csv", help="Print csv once parsed to dataframe to std.out", type=bool, default=False)
    parser.add_argument("-b", "--create_boxplots", help="Whether to create boxplots from csv file or not", type=bool, default=False)
    parser.add_argument("-d", "--create_hists", help="Whether to create histogrammed distributions from csv file or not", type=bool, default=False)
    args=parser.parse_args()
    run_main(args)