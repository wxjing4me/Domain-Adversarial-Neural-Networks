import sys
import time
from data_helper import *
from flip_gradient import flip_gradient
from utils import *
from DANN import *

def main_csv():
    if len(sys.argv) == 1:
        Xs, ys = data_helper.get_data('supernova-src')
        Xt, yt = data_helper.get_data('supernova-tgt')
    else:
        Xs, ys = data_helper.get_data(sys.argv[1])
        Xt, yt = data_helper.get_data(sys.argv[2])

    # train_and_evaluate(op='Domain Classification', X_src=Xs, y_src=ys, X_tgt=Xt, y_tgt=yt, grad_scale=-1.0)
    # train_and_evaluate(op='Label Classification', X_src=Xs, y_src=ys, X_tgt=Xt, y_tgt=yt)
    # train_and_evaluate(op='Domain Adaptation', X_src=Xs, y_src=ys, X_tgt=Xt, y_tgt=yt)
    train_and_evaluate(op='Deep Domain Adaptation', X_src=Xs, y_src=ys, X_tgt=Xt, y_tgt=yt)

def main():
    data_folder = '../data/' # where the datasets are
    source_name = 'books'   # source domain: books, dvd, kitchen, or electronics
    target_name = 'electronics'     # traget domain: books, dvd, kitchen, or electronics

    ## Loading pre-embedded data
    print("Loading data ...")
    Xs, ys, Xt, yt, xtest, ytest = load_amazon(source_name, target_name, data_folder)

    train_and_evaluate(op='Domain Classification', X_src=Xs, y_src=ys, X_tgt=Xt, y_tgt=yt, grad_scale=-1.0)
    train_and_evaluate(op='Label Classification', X_src=Xs, y_src=ys, X_tgt=Xt, y_tgt=yt)
    train_and_evaluate(op='Domain Adaptation', X_src=Xs, y_src=ys, X_tgt=Xt, y_tgt=yt)
    train_and_evaluate(op='Deep Domain Adaptation', X_src=Xs, y_src=ys, X_tgt=Xt, y_tgt=yt)



if __name__ == '__main__':
    main()