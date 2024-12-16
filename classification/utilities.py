import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class SimpleLRScheduler(_LRScheduler):
    def __init__(self, optimizer, learning_rate=0.1, last_epoch=-1, verbose=False):
        """
        Scheduler che imposta un tasso di apprendimento costante.

        Parametri:
        ----------
        optimizer : torch.optim.Optimizer
            Ottimizzatore associato a cui applicare lo scheduler.
        learning_rate : float, opzionale
            Tasso di apprendimento costante (default: 0.1).
        last_epoch : int, opzionale
            Indice dell'ultimo epoch. Usato per riprendere da checkpoint (default: -1).
        verbose : bool, opzionale
            Se True, stampa il tasso di apprendimento a ogni aggiornamento (default: False).
        """
        self.learning_rate = learning_rate
        super(SimpleLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        Restituisce il tasso di apprendimento corrente per ogni gruppo di parametri.
        """
        # Imposta il tasso di apprendimento costante
        return [self.learning_rate for _ in self.optimizer.param_groups]

class prettyfloat(float):
    def __repr__(self):
        return "%0.3f" % self


def get_classweights(att_list_tr):
    """
    Calcola i pesi di classe per un dataset che include 0, 1 e -1 (dati mancanti).

    Parametri:
    ----------
    att_list_tr : np.ndarray
        Array di attributi (num_samples, num_classes) con valori 0, 1 o -1.

    Ritorna:
    --------
    class_weights : np.ndarray
        Array di pesi per ciascuna classe (num_classes,).
    """
    class_weights = np.ones((att_list_tr.shape[1],))  # Inizializza i pesi a 1
    for att in range(att_list_tr.shape[1]):  # Itera su ogni classe
        current_att = att_list_tr[:, att]  # Valori per la classe corrente

        # Filtra gli esempi con valori validi (0 o 1) ed escludi -1
        valid_indices = current_att != -1
        valid_att = current_att[valid_indices]

        # Calcola la proporzione di esempi positivi
        cl_imb_tr = np.sum(valid_att) / float(valid_att.shape[0]) if valid_att.shape[0] > 0 else 0

        # Se la classe è sbilanciata, calcola il peso
        if cl_imb_tr > 0 and cl_imb_tr <= 0.5:
            class_weights[att] = (1 - cl_imb_tr) / float(cl_imb_tr)
    return class_weights.astype("float32")


# def get_data(full_path):
#     all_im_list_tr = np.array(
#         [line.rstrip('\n')[1:-2] for line in open(full_path + 'wider_att/wider_att_train_imglist.txt')])
#     all_att_list_tr = np.array(
#         [map(int, line.rstrip(' \n').split(" ")) for line in open(full_path + 'wider_att/wider_att_train_label.txt')])
#
#     # Split To Train and Validation
#     im_list_tr = []
#     att_list_tr = []
#     im_list_val = []
#     att_list_val = []
#     for im_path, att in zip(all_im_list_tr, all_att_list_tr):
#         if im_path.split("/")[2] == 'val':
#             im_list_val.append(im_path)
#             att_list_val.append(att)
#         else:
#             im_list_tr.append(im_path)
#             att_list_tr.append(att)
#     im_list_test = np.array([line.rstrip('\n')[1:-2] for line in open(full_path + 'wider_att/wider_att_test_imglist.txt')])
#     att_list_test = np.array([map(int, line.rstrip(' \n').split(" ")) for line in open(full_path + 'wider_att/wider_att_test_label.txt')])
#     return np.array(im_list_tr), np.array(att_list_tr), np.array(im_list_val), np.array(att_list_val), np.array(im_list_test), np.array(att_list_test)
#
# def get_iterators(batch_size, num_classes, data_shape):
#     train = ImageRecordIter(
#         path_imgrec='wider_records/training_list.rec',
#         path_imglist='wider_records/training_list.lst',
#         batch_size=batch_size,
#         data_shape=data_shape,
#         preprocess_threads=4,
#         mean_r=104,
#         mean_g=117,
#         mean_b=123,
#         resize=256,
#         max_crop_size=224,
#         min_crop_size=128,
#         label_width=num_classes,
#         shuffle=False,
#         round_batch=False,
#         rand_crop=True,
#         rand_mirror=True)
#     val = ImageRecordIter(
#         path_imgrec='wider_records/valid_list.rec',
#         path_imglist='wider_records/valid_list.lst',
#         shuffle=False,
#         mean_r=104,
#         mean_g=117,
#         mean_b=123,
#         round_batch=False,
#         label_width=num_classes,
#         preprocess_threads=4,
#         batch_size=batch_size,
#         data_shape=data_shape)
#     test = ImageRecordIter(
#         path_imgrec='wider_records/testing_list.rec',
#         path_imglist='wider_records/testing_list.lst',
#         shuffle=False,
#         round_batch=False,
#         mean_r=104,
#         mean_g=117,
#         mean_b=123,
#         label_width=num_classes,
#         preprocess_threads=4,
#         batch_size=batch_size,
#         data_shape=data_shape)
#     return train, val, test
