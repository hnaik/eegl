import logging
from datetime import datetime

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

import egr.v2.filtering as flt
import egr.v2.iso_match as im
import egr.util as util

LOG = logging.getLogger(__name__)


def main(data, args):
    filtered = flt.get_filtered_candidates(data, args)
    annotations = get_annotations(data, filtered['candidates'], args)

    return {
        'filtering': filtered['timing'],
        'annotations': annotations['timing'],
    }


def get_annotations(data, candidates, args):
    assert (
        len(candidates) == data.x.shape[1]
    ), f'#candidates({len(candidates)}) != #features({data.x.shape[1]})'

    begin = datetime.now()

    params = []
    for i, node in enumerate(data.G.nodes()):
        for j in range(data.x.shape[1]):
            params.append((data.G, candidates[j], node, i, j))

    h = torch.zeros(data.x.shape)

    pbar = tqdm(params)
    parallel = Parallel(n_jobs=-2, return_as='generator')
    result = parallel(delayed(im.is_subgraph_match)(*args) for args in pbar)

    for r, c, iso in result:
        h[r, c] = 1.0 if iso else 0.0

    end = datetime.now()
    LOG.info('Finished annotations in %ss', end - begin)
    LOG.info('Saving features %s to %s', h.shape, args.output_feature_file)
    util.save_features(args.output_feature_file, h)

    return {'annotations': h, 'timing': {'begin': begin, 'end': end}}
