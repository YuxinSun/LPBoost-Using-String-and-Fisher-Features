__author__ = 'yuxinsun'
import cPickle


def save_pickle(path, file_name, to_save):
    """
    save to cpickle files
    Parameters
    -------
    :param path: string
        path where file is to be saved
    :param file_name: string
        file name
    :param to_save: variable
        variable to be saved
    """
    with open(path + file_name + '.cpickle', 'wb') as f:
        # pickle.dump(to_save, f, pickle.HIGHEST_PROTOCOL)
        cPickle.dump(to_save, f, cPickle.HIGHEST_PROTOCOL)


def read_pickle(path, file_name):
    """
    Read from cpickle files
    :param path: string
        path
    :param file_name: string
        file name
    """
    with open(path + file_name + '.cpickle', 'rb') as f:
        return cPickle.load(f)