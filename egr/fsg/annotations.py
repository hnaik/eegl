import logging
import pickle
import typing as ty
from datetime import datetime

import networkx as nx

import egr.parallel.annotations as epa
from egr.fsg import filtering
from egr.util import normalize_path, save

LOG = logging.getLogger(__name__)


def main(data, args) -> ty.Dict[str, ty.Dict]:
    LOG.debug('args=%s', args)
    G: nx.Graph = data.G

    filter_begin = datetime.now()
    selected: ty.List[nx.Graph] = (
        filtering.filter_graphs_with_feature_importance(data, args)
    )
    filter_end = datetime.now()

    LOG.info('Finished filtering in %s', filter_end - filter_begin)
    save_intermediate(args, selected)
    explain_ids = (
        data.explain_idx if hasattr(data, 'explain_idx') else G.nodes()
    )
    orig_feat_dim = data.x_orig.shape[1] if hasattr(data, 'x_orig') else 0
    LOG.info(
        'original feature dim: %d, num_features:%d',
        orig_feat_dim,
        data.num_node_features,
    )
    annotation_dim: int = data.x.shape[1] - orig_feat_dim
    annotation_begin = datetime.now()
    epa.perform_annotations(
        G, selected, args, explain_ids, dims=annotation_dim
    )
    annotation_end = datetime.now()

    return {
        'filtering': {'begin': filter_begin, 'end': filter_end},
        'annotation': {'begin': annotation_begin, 'end': annotation_end},
    }


def save_intermediate(args, sg):
    num_sg = len(sg)
    LOG.debug('Generating %d intermediate subgraphs', num_sg)
    inter_dir = normalize_path(args.fsg_dir)
    inter_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_sg):
        path = inter_dir / f'{i:04d}.json'
        label = sg[i].graph['label']
        score = sg[i].graph['f1_score']
        sg[i].graph['feature_index'] = i
        LOG.debug(
            'Saving FSG to %s (label:%d, score:%.4f)',
            path,
            label,
            score,
        )
        save(sg[i], path)

    features_path = inter_dir / 'features.pkl'
    LOG.info('Saving all features to: %s', features_path)
    pickle.dump(sg, features_path.open('wb'))
