import re
import os

def find_matr_files(folder):
    ret = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if is_matr_file(file):
                path = os.path.join(root, file)
                ret.append(path)
    return ret

def is_matr_file(path):
    return re.fullmatch(r'matrk\dm\d.txt', path)

def get_matr_size(path):
    matr = []
    with open(path) as file:
        for line in file.readlines():
            matr.append(list(map(float, line.split())))
    matr = [it for it in matr if it != []]
    height = len(matr)
    width = len(matr[0])
    return height, width

def find_db_folders(folder):
    ret = []
    for root, dirs, file in os.walk(folder):
        for dir_ in dirs:
            if dir_.startswith('DB'):
                ret.append(dir_)
    return ret

def find_set_file(folder):
    for root, dirs, files in os.walk(folder):
        for dir_ in dirs:
            path = os.path.join(root, dir_, folder.split('DB_')[1] + '.set')
            if os.path.exists(path):
                return path

def get_properties(path):
    ret = []
    with open(path) as file:
        for line in file.readlines():
            if line.startswith('IN'):
                ret.append([line.split()[1]])
##            if line.startswith('MF'):
##                ret[-1].append(line.split()[1])
            found_A = re.search(r'A\d+', line)
            if found_A and found_A.start() == 0:
                ret[-1].append(line.split()[1])
    return ret

for db in find_db_folders('.'):
    set_file = find_set_file(db)
    
    with open(os.path.join(db, 'properties.txt'), 'w') as file:
        sizes = []
        for matr_file in find_matr_files(db):
            file_name = os.path.split(matr_file)[1]
            size = get_matr_size(matr_file)
            sizes.append(f'{file_name} - {size}')
        file.write('\n'.join(sizes))

        file.write('\n\n')

        for property_ in get_properties(set_file):
            file.write(' '.join(property_) + '\n')
