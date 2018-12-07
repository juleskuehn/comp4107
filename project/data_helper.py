
import re
import random

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(file_path):
    """
    Load the book titles and the corresponding category of those books, split the book title into word vectors
    Each row in the orignal csv file looks like
    "[AMAZON INDEX (ASIN)}","[FILENAME]","[IMAGE URL]","[TITLE]","[AUTHOR]","[CATEGORY ID]","[CATEGORY]"
    Here we only need the title and category, so we only take those information

    Return split titles and categories
    """

    # some special character cannot be recognized correctly, so adding the error=ignore to ignore those errors
    raw_data = list(open(file_path, "r", errors="ignore").readlines()) # raw data containinng all information
    
    data = []

    for row in raw_data:
        book_info = row.split('","') # split the data
        data.append([clean_str(book_info[3]), book_info[-1][:-1]]) # remove the last '"' symbol from category label

    return data


