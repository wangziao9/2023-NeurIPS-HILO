import numpy as np

def load_gnt(filename):
    "Parses the gnt file according to CASIA's Offline handwriting database's format"
    import struct
    def getInt2(f):
        return struct.unpack('H', f.read(2))[0]
    with open(filename, 'rb') as f:
        labels = []
        arrs = []
        while True:
            ssize_bytes = f.read(4)
            if not ssize_bytes:
                break
            this_field_size = struct.unpack('I', ssize_bytes)[0]
            # Get character's Unicode code point
            tag_code = f.read(2).decode('GB2312', errors='ignore')
            labels.append(ord(tag_code[0]) if tag_code else 0)
            # Get bitmap
            width, height = getInt2(f), getInt2(f)
            arr = np.frombuffer(f.read(height * width), dtype=np.uint8).reshape((height, width))
            arrs.append(arr)
    return arrs, labels

def load_all_gnt(base_dir):
    arrs, labels = [], []
    import os
    for f in os.listdir(base_dir):
        a, l = load_gnt(base_dir+f)
        arrs.extend(a)
        labels.extend(l)
    return arrs, labels

def visualize_gnt():
    "Example visualization of the handwritten characters"
    arrs, labels = load_gnt("../assets/Gnt1.1Test/1241-f.gnt")
    arr = np.expand_dims(255-arrs[316],-1)
    print(labels[316])
    import matplotlib.pyplot as plt
    import tensorflow as tf
    arr = tf.image.resize_with_pad(arr, 100, 100, antialias=True)
    print(arr.shape)
    pad = 2
    arr = np.pad(arr, ((pad, pad), (pad, pad), (0,0)), mode='constant')
    plt.imshow(arr)
    plt.show()

def stat_gnt():
    "Print statistics of the Gnt1.1Test dataset"
    arrs, labels = load_all_gnt("../assets/Gnt1.1Test/")
    print(len(arrs))
    print(len(labels))
    from collections import Counter
    d = Counter(labels)
    print(d["视"])
    print(d["觉"])
    print(len(d))