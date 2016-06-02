import numpy as np


def remain_top_communities(labels, max_clusters=None, min_count=3):
    # Remain top 14 clusters and remove singleton ones.
    unique = set(labels)
    sizes = {}
    for label in unique:
        sizes[label] = np.sum(labels == label)
    top_labels = sorted(sizes, key=lambda item: sizes[item], reverse=True)
    if max_clusters:
        top_labels = top_labels[:max_clusters + 1] if -1 in top_labels else top_labels[:max_clusters]
    if -1 in labels and -1 not in top_labels:
        top_labels.append(-1)
    new_label = 0
    for label in set(labels):
        if label == -1:
            continue
        if label not in top_labels or np.sum(labels == label) < min_count:
            labels[labels == label] = -1
        else:
            if label != new_label:
                labels[labels == label] = new_label
            new_label += 1

    return labels