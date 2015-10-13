import re
import string
import operator
import pandas as pandas
import numpy as np

def substringsLocations_in_string(big_string, substrings):
    for i,substring in enumerate(substrings):
        if not pandas.isnull(big_string) and string.find(big_string, substring) != -1:
            return i
    return -1


def createCategory(data,source,dest,size,max):
    number_of_categories = max // size
    data[dest] = ((data[source]//size)
                      .clip_upper(number_of_categories-1)
                      .astype(np.int))