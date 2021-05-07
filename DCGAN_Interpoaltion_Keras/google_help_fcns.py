# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from lib import data
import numpy as np
import scipy.spatial
from lines import draw_line
#from interpolation import linear_interpol, stochastic_interpol, stochasticSMC_interpol
from interpolations_help_fcns import get_valid_code


def closest_line(query_lines, metric='cosine'):
    """Compute the distance to, and parameters for, the closest line to each
    line in query_lines.
    Args:
        - query_lines: Array of lines to compute closest matches for, shape
            (n_lines, width, height, 1)
        - metric: String to pass to scipy.spatial.distance.cdist to choose
            which distance metric to use
    Returns:
        - min_dist, starts, ends: Arrays of shape (n_lines,) denoting the
            distance to the nearest ``true'' line and the start and end points.
    """
    h, w = query_lines.shape[1:-1]
    # Construct 10000 lines with these dimensions
    angles = np.linspace(0, 2*np.pi - 2*np.pi/10000, 10000)
    all_lines = np.array(
        [(draw_line(angle, h, w)) for angle in angles])
    # Produce vectorized versions of both for use with scipy.spatial
    flat_query = query_lines.reshape(query_lines.shape[0], -1)
    flat_all = all_lines.reshape(all_lines.shape[0], -1)

    # Compute pairwise distance matrix of query lines with all valid lines
    distances = scipy.spatial.distance.cdist(flat_query, flat_all, metric)
    min_dist_idx = np.argmin(distances, axis=-1)
    min_dist = distances[np.arange(distances.shape[0]), min_dist_idx]
    angles = np.array([angles[n] for n in min_dist_idx])
    
    return min_dist, angles


def smoothness_score(angles):
    """Computes the smoothness score of a line interpolation according to the
    angles of each line.
    Args:
        - angles: Array of shape (n_interpolations, n_lines_per_interpolation)
            giving the angle of each line in each interpolation.
    Returns:
        - smoothness_scores: Array of shape (n_interpolations,) giving the
            average smoothness score for all of the provided interpolations.
    """
    angles = np.atleast_2d(angles)
    # Remove discontinuities larger than np.pi
    angles = np.unwrap(angles)
    diffs = np.abs(np.diff(angles, axis=-1))
    # Compute the angle difference from the first and last point
    total_diff = np.abs(angles[:, :1] - angles[:, -1:])
    # When total_diff is zero, there's no way to compute this score
    zero_diff = (total_diff < 1e-4).flatten()
    normalized_diffs = diffs/total_diff
    deviation = np.max(normalized_diffs, axis=-1) - 1./(angles.shape[1] - 1)
    # Set score to NaN when we aren't able to compute it
    deviation[zero_diff] = np.nan
    return deviation


def line_eval(interpolated_lines):
    """Given a group of line interpolations, compute mean nearest line distance
    and mean smoothness score for all of the interpolations.
    This version of this metric is meant for vertical lines only.
    Args:
        - interpolated_lines: Collection of line interpolation images, shape
            (n_interpolations, n_lines_per_interpolation, height, width, 1)
    Returns:
        - mean_distance: Average distance to closest ``real'' line.
        - mean_smoothness: Average interpolation smoothness
    """
    original_shape = interpolated_lines.shape
    min_dist, angles = closest_line(
        interpolated_lines.reshape((-1,) + original_shape[2:]))
    mean_distance = np.mean(min_dist)
    
    smoothness_scores = smoothness_score(
        angles.reshape(original_shape[0], original_shape[1]))
    nan_scores = np.isnan(smoothness_scores)
    # If all scores were NaN, set the mean score to NaN
    if np.all(nan_scores):
        mean_smoothness = np.nan
    # Otherwise only compute mean for non-NaN scores
    else:
        sum_smoothness = np.sum(smoothness_scores[np.logical_not(nan_scores)])
        mean_smoothness = sum_smoothness/float(len(nan_scores))
    return np.float32(mean_distance), np.float32(mean_smoothness)


def metrics(G,D,GAN,config):

    k = config.metrics_k

    mean_dist_vec = np.zeros((k,1))
    mean_smoothness_vec = np.zeros((k,1))

    print('')
    print('Calculating {} metrics...'.format(config.metrics_type))
    print('')

    for i in range(k):

        print('Interpolation {}/{}.'.format(i+1,config.metrics_k))

        #z0 = np.array([[],[]])
        #zT = np.array([[],[]])

        z0 = get_valid_code(GAN, config)
        z0 = (z0/np.linalg.norm(z0))*np.sqrt(config.z_dim)

        zT = get_valid_code(GAN, config)
        zT = (zT/np.linalg.norm(zT))*np.sqrt(config.z_dim)

        if config.metrics_type == 'linear':
            path,_,_ = linear_interpol(config,z0,zT)
        elif config.metrics_type == 'stoch':
            path,_,_ = stochastic_interpol(G,D,GAN,config,z0,zT)
        elif config.metrics_type == 'stochSMC':
            path,_,_ = stochasticSMC_interpol(G,D,GAN,config,z0,zT)
        

        path_vis = G.predict(path.T)
        
        mean_dist,mean_smoothness = line_eval(path_vis[np.newaxis,:])

        mean_dist_vec[i] = mean_dist
        mean_smoothness_vec[i] = mean_smoothness
        

    mean_dist = np.round(np.mean(mean_dist_vec),8)
    mean_smoothness = np.round(np.mean(mean_smoothness_vec),8)
    std_dist = np.round(np.std(mean_dist_vec,0),8)
    std_smoothness = np.round(np.std(mean_smoothness_vec),8)

    min_dist = np.round(np.min(mean_dist_vec),8)
    min_smoothness = np.round(np.min(mean_smoothness_vec,0),8)

    print('')
    print('Mean dist: {}, Std dist: {}'.format(mean_dist,std_dist))
    print('Mean smoothness: {}, Std smoothness: {}'.format(mean_smoothness,std_smoothness))
    #print('Min dist: {}'.format(min_dist))
    #print('Min smoothness: {}'.format(min_smoothness))


    #print(mean_smoothness_vec)