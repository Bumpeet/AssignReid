from numpy.linalg import norm
import numpy as np

def cosine(A, B):

  # compute cosine similarity
  cosine = np.dot(A,B)/(norm(A)*norm(B))
  return cosine

def map_outer(outer: list[list[str]], imgs: list[str]) -> list[list[str]]:
  '''
  This method forms a list containning inner list with the names that obtained from cluster
  '''
  outer_names = []
  for inner in outer:
    inner_names = []
    for i in inner:
      inner_names.append(imgs[i])
    outer_names.append(inner_names)

  return outer_names

def generate_batches(imgs: list[str], bs: int) -> np.ndarray:
    '''
    This method helps in generating the list of images into batches

    imgs: contains the list of names of all the images
    bs: batch size

    returns the np.ndarray which contains the names in the shape of (n_batches, bs)
    '''
    n_batches = len(imgs)//bs
    imgs = np.array(imgs[:n_batches*bs]).reshape(n_batches, bs)
    return imgs