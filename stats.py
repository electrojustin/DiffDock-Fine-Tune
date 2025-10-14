import glob
import os
import numpy as np

results_dir = os.environ["RESULTS_DIR"] + '/'
num_shards=len(list(glob.glob(results_dir + 'run_times_*.npy')))
total = 0
any_rmsd = 0
all_rmsd = 0
top1_rmsd = 0
top5_rmsd = 0
any_centroid = 0
all_centroid = 0
top1_centroid = 0
top5_centroid = 0
for i in range(0, num_shards):
    complex_names = np.load(results_dir + 'complex_names_' + str(i) + '.npy')
    confidences = np.load(results_dir + 'confidences_' + str(i) + '.npy')
    centroids = np.load(results_dir + 'centroid_distances_' + str(i) + '.npy')
    rmsds = np.load(results_dir + 'rmsds_' + str(i) + '.npy')
    for j in range(0, len(complex_names)):
        total += 1
        rankings = np.argsort(confidences[j, :])[::-1]
        any_rmsd += np.any(rmsds[j, :] < 2.0)
        all_rmsd += np.sum(rmsds[j, :] < 2.0) / rmsds.shape[1]
        top1_rmsd += rmsds[j, :][rankings[0]] < 2.0
        top5_rmsd += np.any(rmsds[j, :][rankings[0:5]] < 2.0)
        any_centroid += np.any(centroids[j, :] < 2.0)
        all_centroid += np.sum(centroids[j, :] < 2.0) / centroids.shape[1]
        top1_centroid += centroids[j, :][rankings[0]] < 2.0
        top5_centroid += np.any(centroids[j, :][rankings[0:5]] < 2.0)

print('Total number of complexes: ' + str(total))
print('any_rmsd: ' + str(any_rmsd / total * 100))
print('all_rmsd: ' + str(all_rmsd / total * 100))
print('top1_rmsd: ' + str(top1_rmsd / total * 100))
print('top5_rmsd: ' + str(top5_rmsd / total * 100))
print('any_centroid: ' + str(any_centroid / total * 100))
print('all_centroid: ' + str(all_centroid / total * 100))
print('top1_centroid: ' + str(top1_centroid / total * 100))
print('top5_centroid: ' + str(top5_centroid / total * 100))
