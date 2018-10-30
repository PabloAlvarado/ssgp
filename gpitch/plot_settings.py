import matplotlib
if True: matplotlib.use('agg') # define if running code on server (True)
from matplotlib import pyplot as plt


def plot_init_settings(visible_device='0', interactive=False):
    '''
    plotting initial configuration
    '''
    plt.rcParams['figure.figsize'] = (18, 6)  # set plot size
    plt.interactive(True)
    plt.close('all')


def myplot():
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.115, bottom=.12, right=.99, top=.97)
    fig.set_size_inches(width, height)
    plt.grid(False)





































#
