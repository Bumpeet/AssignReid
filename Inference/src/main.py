import os
from glob import glob
from model import RadiusRequirement
from argparse import ArgumentParser


def main(thresh, bs):
    weights = './models/osnet_x0_25_msmt17.pt'
    model_name = 'osnet_x0_25'

    try:
        os.remove('./data/.DS_Store')
    except:
        print('file cannot be removed')    

    try:
        os.remove('./data/3830.jpg') 
    except:
        print('file cannot be removed')

    imgs = glob('./data/*')[:]
    n_batches = len(imgs)//bs # total number of batches

    RR = RadiusRequirement(model_name, weights, imgs, bs, n_batches=n_batches, thresh=thresh)

    RR.Task1()
    print("\t\t\t\t ================ Done Task-1 ================")

    RR.Task2()
    print("\t\t\t\t ================ Done Task-2 ================")

    RR.Task3()
    print("\t\t\t\t ================ Done Task-3 ================")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--t', type=float, default=.75, help="threshold to match a person based on cosine similarity (0 to 1)")
    parser.add_argument('--bs', type = int, default=32, help="batch size to run the inference")
    # parser.add_argument('--n', type = int, default=10, help= "number of unique persons embedding to plot on 2D space")
    args = parser.parse_args()

    thresh = args.t
    bs = args.bs
    # n = args.n
    main(thresh, bs)
