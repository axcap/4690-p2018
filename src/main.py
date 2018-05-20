import utils as utils

def createGenerator():
  mylist = range(3)
  for i in mylist:
    yield i*i



if __name__ == "__main__":

  stream = utils.frameGenerator() # create a generator
  for frame in stream:
    print(i)
