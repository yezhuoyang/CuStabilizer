# local imports
import sys

sys.path.append(r"/home/yezhuoyang98_g_ucla_edu/John/CuStabilizer/")
from tableau import tableau
import os
import re


def list_files_in_directory(directory):
    try:
        # List all files and directories in the specified directory
        all_files = os.listdir(directory)
        # Filter out directories, keeping only files
        files = [f for f in all_files if os.path.isfile(os.path.join(directory, f))]
        return files
    except Exception as e:
        return str(e)



def get_largest_integer_from_file(filepath):
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            
        # Find all integers in the file using regular expression
        numbers = re.findall(r'\d+', content)
        
        # Convert the found strings to integers
        int_numbers = [int(num) for num in numbers]
        
        # Find and return the largest integer
        if int_numbers:
            return max(int_numbers)
        else:
            return None  # In case there are no numbers in the file
    except Exception as e:
        return str(e)



if __name__=="__main__":
    directoryname="/home/yezhuoyang98_g_ucla_edu/John/CuStabilizer/testcases/"
    filelist=list_files_in_directory(directoryname)
    
    for file in filelist:
        fullpath=directoryname+file
        qubitnum=get_largest_integer_from_file(fullpath)
        Tb=tableau(qubitnum+1)
        Tb.init_tableau()
        Tb.read_instructions_from_file(fullpath)
        Tb.calculate()
    
    