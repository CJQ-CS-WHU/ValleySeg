def dataset_list_provider(rate):
    i = 0
    leave = rate * 10
    with open(r'F:\ValleySeg\datasets\valid.txt', 'r') as valid:
        with open(r'F:\ValleySeg\datasets\train.txt', 'w') as train:
            with open(r'F:\ValleySeg\datasets\test.txt', 'w') as test:
                line = valid.readline()
                while not line == '':

                    if i % 10 >= leave:
                        test.write(line)
                    else:
                        train.write(line)
                    line = valid.readline()
                    i += 1
    print('devide finished')


def valid_datasets_provider():
    with open(r'F:\ValleySeg\datasets\dataset_list_total.txt', 'r') as dataset_list:
        with open(r'F:\ValleySeg\datasets\valid.txt', 'w') as valid_list:
            line = dataset_list.readline()
            while not line == '':
                name = line.split(':')[1].split('\t')[0]
                valid = line.split(':')[-1]
                if valid == '1\n':
                    valid_list.write(name + '\n')
                line = dataset_list.readline()
    print('build finished')


if __name__ == '__main__':
    # valid_datasets_provider()
    dataset_list_provider(0.8)
