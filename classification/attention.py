import numpy as np
from models import get_conv2D, get_fatt, get_fsr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_action_labels(path_list):
    """
    Estrae le azioni dai percorsi forniti, le mappa a etichette numeriche e restituisce un array NumPy di tali etichette.

    Parametri:
    ----------
    path_list : list
        Lista di stringhe, ciascuna rappresenta un percorso di file.

    Ritorna:
    --------
    np.array
        Array NumPy contenente le etichette numeriche corrispondenti alle azioni estratte dai percorsi.
    """
    # Inizializza una lista per salvare le azioni estratte dai percorsi
    action_list = []

    # Estrai l'azione da ogni percorso
    for p in path_list:
        # Divide il percorso usando "/" come separatore e seleziona il quarto elemento (indice 3)
        # Poi divide ulteriormente l'elemento selezionato usando "--" e prende il secondo elemento (indice 1)
        action_list.append(p.split("/")[3].split("--")[1])

    # Crea un dizionario che assegna un'etichetta numerica unica a ciascuna azione
    # `set(action_list)` rimuove i duplicati per ottenere un insieme unico di azioni
    # `enumerate` assegna un indice numerico a ciascuna azione
    d = {key: value for (value, key) in enumerate(set(action_list))}

    # Inizializza una lista per salvare le etichette numeriche delle azioni
    actions = []

    # Converti ciascuna azione nella sua etichetta numerica utilizzando il dizionario `d`
    for action in action_list:
        actions.append(d[action])

    # Converte la lista di etichette numeriche in un array NumPy e la restituisce
    return np.array(actions)


def compute_attention(features, fconv, fatt):
    """
    Calcola la rappresentazione convoluzionale e una mappa di attenzione spaziale normalizzata.

    Parametri:
    ----------
    features : torch.Tensor
        Tensore di input con dimensioni tipiche (batch_size, num_channels, height, width).
    fconv : nn.Module
        Modello o funzione convoluzionale applicata a `features`.
    fatt : nn.Module
        Modello o funzione per calcolare i pesi di attenzione spaziale da `features`.

    Ritorna:
    --------
    output_conv : torch.Tensor
        Risultato della convoluzione tramite `fconv(features)`.
    spatial_softmax : torch.Tensor
        Mappa di attenzione spaziale normalizzata tramite softmax, con stessa forma di `features`.
    """
    # 1. Applica fconv per calcolare una trasformazione convoluzionale delle caratteristiche
    output_conv = fconv(features)

    # 2. Applica fatt per calcolare i pesi di attenzione spaziale non normalizzati
    output_att = fatt(features)

    # 3. Rimodella per appiattire le dimensioni spaziali
    temp_f = output_att.reshape(output_att.shape[0] * output_att.shape[1], output_att.shape[2] * output_att.shape[3])
    # 4. Calcola la softmax lungo la dimensione spaziale
    spatial_softmax = F.softmax(temp_f,dim=1).reshape(output_att.shape[0], output_att.shape[1], output_att.shape[2], output_att.shape[3])
    return output_conv, spatial_softmax

def attention_net_trainer(lr_scheduler, classes, args, stride, ctx):
    """
        Configura modelli e ottimizzatori per una rete di attenzione spaziale in PyTorch.

        Parametri:
        ----------
        lr_scheduler : torch.optim.lr_scheduler
            Pianificatore del tasso di apprendimento per gli ottimizzatori.
        classes : int
            Numero di classi del modello.
        args : Namespace
            Argomenti contenenti i parametri di addestramento, come:
                - args.test: Booleano per indicare la modalità test.
                - args.mom: Momentum per l'ottimizzatore.
                - args.wd: Weight decay (regolarizzazione L2).
        stride : int
            Passo (stride) per le convoluzioni.
        ctx : torch.device
            Dispositivo su cui allocare i modelli (es. torch.device("cuda") o "cpu").

        Ritorna:
        --------
        fconv_stg : nn.Module
            Modello convoluzionale per l'elaborazione delle caratteristiche.
        fatt_stg : nn.Module
            Modello per generare mappe di attenzione spaziale.
        trainer_conv : optim.Optimizer
            Ottimizzatore per il modello convoluzionale (vuoto in modalità test).
        trainer_att : optim.Optimizer
            Ottimizzatore per il modello di attenzione (vuoto in modalità test).
        """
    # 1. Crea il modello convoluzionale
    fconv_stg = get_conv2D(classes, stride, ctx)

    # 2. Crea il modello per l'attenzione spaziale
    fatt_stg = get_fatt(classes, stride, ctx)

    # 3. Inizializza liste vuote per gli ottimizzatori
    trainer_conv, trainer_att = None, None

    # 4. Se non siamo in modalità test, configuriamo gli ottimizzatori
    if not args.test:

        # Configura l'ottimizzatore per il modello convoluzionale
        trainer_conv = optim.SGD(
            fconv_stg.parameters(),  # Parametri del modello
            lr=lr_scheduler.learning_rate,  # Usa il tasso di apprendimento configurato nello scheduler
            momentum=args.mom,  # Momentum
            weight_decay=args.wd  # Penalizzazione L2
        )

        # Configura l'ottimizzatore per il modello di attenzione
        trainer_att = optim.SGD(
            fatt_stg.parameters(),
            lr=lr_scheduler.learning_rate,  # Usa il tasso di apprendimento configurato nello scheduler
            momentum=args.mom,
            weight_decay=args.wd
        )

    # 5. Restituisci i modelli e gli ottimizzatori
    return fconv_stg, fatt_stg, trainer_conv, trainer_att


def attention_cl(lr_scheduler, args, ctx, kernel_size=14):
    """
        Configura il modello e l'ottimizzatore per l'attenzione spaziale in PyTorch.

        Parametri:
        ----------
        lr_scheduler : torch.optim.lr_scheduler
            Pianificatore del tasso di apprendimento per l'ottimizzatore.
        args : Namespace
            Argomenti contenenti i parametri di addestramento, come:
                - args.num_classes: Numero di classi.
                - args.mom: Momentum per l'ottimizzatore.
                - args.wd: Weight decay (regolarizzazione L2).
        ctx : torch.device
            Dispositivo su cui allocare il modello (es. torch.device("cuda") o "cpu").
        kernel_size : int, opzionale
            Dimensione del kernel per il modello (default: 14).

        Ritorna:
        --------
        fsr_stg : nn.Module
            Modello convoluzionale.
        trainer_sr : optim.Optimizer
            Ottimizzatore configurato per il modello.
        """
    # 1. Crea il modello convoluzionale
    fsr_stg = get_fsr(args.num_classes, ctx, kernel_size)
    fsr_stg = fsr_stg.to(ctx)  # Sposta il modello sul dispositivo specificato

    # 2. Configura l'ottimizzatore
    trainer_sr = optim.SGD(
        fsr_stg.parameters(),  # Parametri del modello
        lr=lr_scheduler.learning_rate,  # Tasso di apprendimento configurato nello scheduler
        momentum=args.mom,  # Momentum
        weight_decay=args.wd  # Penalizzazione L2
    )

    # 3. Restituisci il modello e l'ottimizzatore
    return fsr_stg, trainer_sr