from lanenet import LaneNet

INPUT_SHAPE = (480, 640, 3)
NUM_CLASSES = 3
EMBEDDING_DIM = 3

lanenet = LaneNet(INPUT_SHAPE, NUM_CLASSES, EMBEDDING_DIM)

print(lanenet.summary())