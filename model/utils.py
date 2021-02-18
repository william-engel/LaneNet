import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth


def create_instance_masks(embeddings, binary_masks, max_num_lanes = 5, min_pixels = 15, min_bin_freq = 1, bandwidth = None):

    B, H, W, E = embeddings.shape # (batch, height, width, embedding dim)
    time_taken = [] # stop time

    instance_masks = []
    for embedding, binary_mask in zip(embeddings, binary_masks):
        instance_mask = np.zeros([H, W], dtype=np.uint8) 

        embedding = embedding[binary_mask != 0] # remove backround

        if bandwidth = None:
            bandwidth = estimate_bandwidth(embedding, quantile=0.2, n_samples=100)
        
        ms = MeanShift(bandwidth = bandwidth, bin_seeding = True, min_bin_freq = min_bin_freq)

        if len(embedding) != 0:

            start_time = time.time() # start time
            labels = ms.fit(embedding).labels_
            time_taken += [time.time() - start_time] # stop time

            instance_mask[binary_mask != 0] = labels + 1

            # remove to small lane instances
            for id in np.unique(instance_mask):
                if len(instance_mask == id) < min_pixels:
                    instance_mask[instance_mask == id] = 0
            
            # remove all lanes aboves max_num_lanes
            ids, cnt = np.unique(instance_mask, return_counts=True)

            if len(ids) > max_num_lanes:
                sindex = np.argsort(cnt)
                for id in ids[sindex[max_num_lanes:]]:
                    instance_mask[instance_mask == id] = 0
    
        instance_masks.append(instance_mask)

    time_taken = np.mean(time_taken)
    instance_masks = np.asarray(instance_masks)

    return instance_masks, time_taken