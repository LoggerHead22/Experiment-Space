from generate_files import *

def create_folder(folder):
    try:
        os.makedirs(folder)
    except:
        pass

def read_matr(path):
    matr = []
    with open(path) as file:
        for line in file.readlines():
            matr.append(list(map(float, line.split())))
    matr = [it for it in matr if it != []]
    return matr

def get_column(m, col):
    ret = []
    for i in range(len(m)):
        ret.append(m[i][col])
    return ret

def prod(m, col1, col2):
    first_col = get_column(m, col1)
    second_col = get_column(m, col2)
    return list(map(lambda x: x[0] * x[1], zip(first_col, second_col)))

def prod_matr(m):
    ret = []
    for i in range(len(m[0])):
        for k in range(i, len(m[0])):
            if i == k:
                tmp = []
                for l in range(len(m)):
                    tmp.append(l * (l - 1) / 2)
                ret += [tmp]
            else:
                ret.append(prod(m, i, k))
    return ret

if __name__ == '__main__':
    for db in find_db_folders('.'):
        matr_files = find_matr_files(db)
        matr_files = [it for it in matr_files if 'PairMatrs' not in it]
        for matr_file in matr_files:
            print(matr_file)
            m = read_matr(matr_file)
            p = prod_matr(m)
            p = list(map(list, zip(*p)))
            file_name = os.path.split(matr_file)[1]
            create_folder(os.path.join(db, 'PairMatrs'))
            with open(os.path.join(db, 'PairMatrs', file_name), 'w') as file:
                for it in p:
                    print(' '.join(map(str, it)), file=file)
