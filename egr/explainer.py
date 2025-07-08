import copy
import logging
import typing as ty

import networkx as nx
import numpy as np
import torch
import torch_geometric.explain as ex
import torch_geometric.data as pygdata
from tqdm import tqdm

import egr.util as eu

LOG = logging.getLogger(__name__)
MAX_SAMPLES = 1000


def explain(cfg, args, data, model, **kw):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = model.to(device)
    explainer = ex.Explainer(
        model=model,
        algorithm=ex.GNNExplainer(epochs=args.num_epochs),
        explanation_type='model',
        edge_mask_type='object',
        node_mask_type='attributes',  # for feature importance
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    def run_explainer(node_id):
        explanation = explainer(data.x, data.edge_index, index=node_id)
        return node_id, explanation

    explain_config = cfg.explain_cfg(**kw)

    nodes_to_explain = select_nodes(data, args.max_explanations)

    pbar = tqdm(nodes_to_explain)

    LOG.info('Generating explanations for %d nodes', len(nodes_to_explain))
    explanations = [run_explainer(node_id) for node_id in pbar]
    LOG.info('Finished generating explanations')

    LOG.info('Writing explanations to: %s', explain_config.explain_dir)

    feature_imp = torch.ones(data.y.shape[0], data.num_node_features)
    for node_id, expl in explanations:
        params = explain_config.explain_input(node_id)
        kw = copy.deepcopy(args)
        kw.__dict__.update(**params)
        feature_imp[node_id, :] = make_importance_features(expl, node_id)
        G, root = get_rooted_subgraph(data.G, data.edge_index, expl, kw)

        G.graph.update(
            {
                'node_idx': kw.explain_node,
                'label': data.y[kw.explain_node].item(),
                '__root__': root,
            }
        )
        G.nodes[root]['__root__'] = 1
        for n in G.nodes():
            if '__root__' in G.nodes()[n] and G.nodes()[n]['__root__'] == 1:
                G.nodes()[n]['label'] = data.y[node_id]
        path = kw.output_root / f'subgraph-{kw.explain_node:04}.json'
        eu.save(G, path)
    feature_imp_path = explain_config.explain_dir / 'feature_importance.pt'
    LOG.info('Saving feature importance to: %s', feature_imp_path)
    torch.save(feature_imp, feature_imp_path)


def select_nodes(data, frac: float):
    num_nodes = data.num_nodes
    indices = (
        data.explain_idx if hasattr(data, 'explain_idx') else range(num_nodes)
    )
    if frac == 1.0:
        return indices
    if frac >= 1:
        frac /= num_nodes

    if frac >= 1:
        return list(range(num_nodes))

    return make_node_samples(data.y, frac)


def make_node_samples(data: pygdata.Data, frac: float) -> ty.List[int]:
    labels = data.y.tolist()

    label_data: ty.Dict = {}
    for i, label in enumerate(labels):
        if label not in label_data:
            label_data.update({label: []})
        label_data[label].append(i)

    rnd = np.random.default_rng()
    sample: ty.List = []
    for label, indices in label_data.items():
        sample.extend(
            rnd.choice(indices, int(frac * len(indices)) + 1, replace=False)
        )
    return sample


def make_importance_features(
    expl, node_id: ty.Optional[int] = None
) -> torch.Tensor:
    node_mask = expl.get('node_mask')
    if node_mask is None:
        raise ValueError('Node masks not available')
    if node_mask.dim() != 2 or node_mask.size(1) <= 1:
        raise ValueError(f'Cannot compute feature importance from {node_mask}')
    return node_mask.sum(dim=0) if node_id is None else node_mask[node_id, :]


# def get_rooted_subgraph(G: nx.Graph, edge_index, explanation, args):
#     em = explanation.edge_mask
#     if em.nonzero(as_tuple=False).size(0) > args.denoise_threshold:
#         em[torch.argsort(em, descending=True)[args.denoise_threshold :]] = 0.0

#     indices = edge_index[:, em.nonzero(as_tuple=False).view(-1).tolist()]
#     edges = indices.t().tolist()
#     edges = [tuple(edge) for edge in edges]

#     H = G.edge_subgraph(edges).copy()
#     if args.explain_node not in H.nodes:
#         H.add_node(args.explain_node)
#     H = H.subgraph(nx.node_connected_component(H, args.explain_node)).copy()
#     H = nx.convert_node_labels_to_integers(H, label_attribute='original')

#     root = [
#         id
#         for id, data in H.nodes(data=True)
#         if data['original'] == args.explain_node
#     ][0]
#     return H, root


def get_rooted_subgraph(G: nx.Graph, edge_index, explanation, args):
    edge_mask = explanation.edge_mask

    # denoise explanation edge mask by only keeping denoise_th largest entries
    if edge_mask.nonzero(as_tuple=False).size(0) > args.denoise_threshold:
        # set all but denoise_th largest entries to 0
        edge_mask[
            torch.argsort(edge_mask, descending=True)[args.denoise_threshold :]
        ] = 0.0

    # get all pairs of edges that are part of the explanation
    edges = (
        edge_index[:, edge_mask.nonzero(as_tuple=False).view(-1).tolist()]
        .t()
        .tolist()
    )
    # edges: list of lists, want list of tuples:
    edges = [tuple(edge) for edge in edges]

    # generate subgraph from edges and only take connected component of node node_id
    sg = G.edge_subgraph(edges).copy()
    if args.explain_node not in sg.nodes:
        sg.add_node(args.explain_node)
    sg = sg.subgraph(nx.node_connected_component(sg, args.explain_node)).copy()

    # relabel nodes to 0,...,n-1 (for gaston) but keep original node_ids as attributes
    sg = nx.convert_node_labels_to_integers(sg, label_attribute='original')
    # find new node_id of root node:
    root = [
        id
        for id, data in sg.nodes(data=True)
        if data['original'] == args.explain_node
    ][0]

    return sg, root
