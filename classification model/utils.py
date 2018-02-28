import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import yaml

def shuflle_data (x,y):
    n= x.shape[0]
    i1 = [i for i in range(n)]
    np.random.shuffle(i1)
    return x[i1],y[i1]


def read_args():
    # read arguments from txt file and give it to model
    args = {}
    with open("parameters.yaml", 'r') as f:
        args.update(yaml.load(f))

        # args['exp']=int(f.readline().split('=')[1].split('\n')[0])
        # args['model']=f.readline().split('=')[1].split('\n')[0]
        # args['num_epochs']=int(f.readline().split('=')[1].split('\n')[0])
        # args['batch_size']=int(f.readline().split('=')[1].split('\n')[0])
        # args['learning_rate']=float(f.readline().split('=')[1].split('\n')[0])
        # args['reg']=float(f.readline().split('=')[1].split('\n')[0])
        # args['dropout']=float(f.readline().split('=')[1].split('\n')[0])
        # args['data_dir']=(f.readline().split('=')[1].split('\n')[0])
        #
        args['exp_dir'] = os.path.realpath(os.getcwd()) + "/" + args['model'] + "/exp" + str(args['exp']) + "/"
        args['checkpoint_dir'] = args['exp_dir'] + "checkpoints/"
        args['summ_dir'] = args['exp_dir'] + "summaries/"

    print('--------------------experiment arguments------------------')

    for k in args.keys():
        print("{}: {}".format(k, args[k]))
        # s.write("{}: {}".format(k, args[k]))


    print("---------------------------------------------------------\n")
    return args


def create_directories(args):
    # read arguments fed to model
    # create directory: experiment: {summary-checkpoints}
    exp_dir = args['exp_dir']
    checkpoint_dir = args['checkpoint_dir']
    summ_dir = args['summ_dir']
    dirs = [exp_dir, summ_dir, checkpoint_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

    print('dumping args file')
    s = open(args['exp_dir'] + 'arguments.txt', 'w+')
    for k in args.keys():
        # print("{}: {}".format(k, args[k]))
        s.write("{}: {}\n".format(k, args[k]))

    s.close()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



# c = read_args()
# create_directories(c)
