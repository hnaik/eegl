import logging
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import torch_geometric.explain as ex

import egr.classifier as clf

import egr.data.pyg_data_helpers as pdh
import egr.result as result

LOG = logging.getLogger(__name__)
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'


class ExplanationBasedAutoEncoder:
    def __call__(self, input_spec):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        params = input_spec.dataset_cfg['params']
        dataset = pdh.get_data_class(input_spec.dataset_cfg)(**params)
        data = pdh.process_data(input_spec, input_spec.fold, {}, dataset[0])
        model: ty.Callable = FeatureEncoder(data, input_spec).to(device)
        opt = optim.Adam(model.parameters(), lr=0.01)

        res = result.Result(
            train_index=data.train_index,
            val_index=data.val_index,
            test_index=data.test_index,
            eval_args={
                'average_strategy': input_spec.average_strategy,
                'cm_normalize': input_spec.confusion_matrix_normalize,
            },
        )

        model.train()
        for epoch in range(5):
            model.zero_grad()
            opt.zero_grad()

            ypred = model(data.x, data.edge_index)
            loss = F.nll_loss(ypred[data.train_mask], data.y[data.train_mask])
            loss.backward()

            opt.step()

            desc = res.compute_train(
                ypred.unsqueeze(0), data.y.unsqueeze(0), loss, 0
            )
            LOG.info('[Epoch:%2d]%s', epoch + 1, desc)

        model.eval()
        y_final = model(data.x, data.edge_index)
        test_desc = res.compute_test(y_final.unsqueeze(0), data.y.unsqueeze(0))
        LOG.info('Test: %s', result.format_result(test_desc))


class FeatureEncoder(nn.Module):
    def __init__(self, data, args):
        super().__init__()

        self._clf = clf.Classifier(data, args)
        self._exp = ex.Explainer(
            self._clf.model,
            algorithm=ex.GNNExplainer(epochs=100),
            explanation_type='model',
            edge_mask_type='object',
            node_mask_type='attributes',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            ),
        )

        std = 0.1
        self.mask = nn.Parameter(torch.rand(data.x.shape) * std)
        self.num_classes = data.num_classes

    def forward(self, x, edge_index):
        x = x.clone()
        with torch.no_grad():
            x = x * self.mask.sigmoid()
        self._clf.train()
        self._clf.model.eval()

        y_expl = torch.FloatTensor(x.shape[0], self.num_classes)
        pbar = tqdm(range(x.shape[0]), bar_format=BAR_FORMAT)
        for node_id in pbar:
            explanation = self._exp(x, edge_index, index=node_id)
            y_ = self._exp.model(explanation.x, explanation.edge_index)
            y_expl[node_id] = y_[node_id]
            pbar.set_description(f'Node ID: {node_id}')
        return y_expl
