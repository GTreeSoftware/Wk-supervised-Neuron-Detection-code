"""
some usually used tools for training and validation
"""
import os
import shutil
import csv

def make_dirs(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)

def remake_dirs(pathname):
    if os.path.exists(pathname):
        shutil.rmtree(pathname)
    os.makedirs(pathname)

def load_file():
    # get current directory path
    current_path = os.path.abspath(__file__)
    # get current parent directory path
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    # path of config.ini file
    config_file_path=os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),'config.ini')
    print('current directory:' + current_path)
    print('current parent directory:' + father_path)
    print('config.ini path:' + config_file_path)


# wirte a list to a csv file
def list2csv(list, file, mode='a+'):
    with open(file, mode) as f:
        w=csv.writer(f)
        w.writerow(list)


def csv_write(out_filename, in_header_list, in_val_list):
    with open(out_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(in_header_list)
        writer.writerows(zip(*in_val_list))


