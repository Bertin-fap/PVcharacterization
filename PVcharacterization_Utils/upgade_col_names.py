def upgrade_col_names(corpuses_folder):
    
    '''Add names to the colummn of the parsing and filter_<i> files to take into account the
    upgrage of BiblioAnalysis_Utils.
    
    Args:
        corpuses_folder (str): folder containing all the corpuses
    '''
    import os
    import pandas as pd
    COL_NAMES = bau.COL_NAMES
    
    # Beware: the new file authorsinst.dat is not present in the Iona's parsing folders
    dict_filename_conversion  = {'addresses.dat':'address',
                                'articles.dat': 'articles',
                                'authors.dat':'authors',
                                'authorsinst.dat':'auth_inst',
                                'authorskeywords.dat':'authorskeywords',
                                'countries.dat':'country',
                                'institutions.dat':'institution',
                                'journalkeywords.dat':'journalkeywords',
                                'keywords.dat':'keywords',
                                'references.dat':'references',
                                'subjects.dat': 'subject',
                                'subjects2.dat':'sub_subject',
                                'titlekeywords.dat':'titlekeywords'}

    new_cols ={ 'authorskeywords.dat': ['Pub_id','Keyword'] ,
                'journalkeywords.dat': ['Pub_id','Keyword'] ,
                'titlekeywords.dat': ['Pub_id','Keyword']}

    dict_filename_conversion = {**dict_filename_conversion , **new_cols} # to be replace by dict_filename_conversion | new_cols
                                                                         # with python 3.9

    for dirpath, dirs, files in os.walk(corpuses_folder):  
        if ('parsing' in   dirpath) |  ('filter_' in  dirpath):
            for file in  [file for file in files if (file.split('.')[1]=='dat') and (file != 'database.dat')]:
                try:
                    print(file, COL_NAMES[dict_filename_conversion[file]],os.path.join(dirpath,file))
                    df = pd.read_csv(os.path.join(dirpath,file),sep='\t',header=None)
                    df.columns = COL_NAMES[dict_filename_conversion[file]]
                    pd.to_csv(os.path.join(dirpath,file),sep='\t')

                except:
                    print('ERROR ',file)
                