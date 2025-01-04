import os.path as osp
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array
import random
import numpy as np
import scipy.sparse as sp

def read_file(folder, name, dtype=None):
    path = osp.join(folder, '{}.txt'.format(name))
    return read_txt_array(path, sep=',', dtype=dtype)

def split(data, batch):
    """
    PyG util code to create graph batches
    """
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    
    # Add root_index slices
    slices['root_index'] = node_slice
    slices['BU_edge_index'] = edge_slice

    return data, slices

def read_graph_data(folder, feature):
    """
    Modified PyG util code to create PyG data instance with root indices
    """
    node_attributes = sp.load_npz(folder + f'new_{feature}_feature.npz')
    edge_index = read_file(folder, 'A', torch.long).t()
    node_graph_id = np.load(folder + 'node_graph_id.npy')
    graph_labels = np.load(folder + 'graph_labels.npy')

    edge_attr = None
    x = torch.from_numpy(node_attributes.todense()).to(torch.float)
    node_graph_id = torch.from_numpy(node_graph_id).to(torch.long)
    y = torch.from_numpy(graph_labels).to(torch.long)
    _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)

    # Create reversed edges for bottom-up processing
    BU_edge_index = edge_index.flip(0)

    # Create root indices (assuming first node of each graph is root)
    unique_graph_ids = torch.unique(node_graph_id)
    root_indices = []
    for graph_id in unique_graph_ids:
        graph_nodes = (node_graph_id == graph_id).nonzero(as_tuple=True)[0]
        root_indices.append(graph_nodes[0])
    root_index = torch.stack(root_indices)

    # Create the data object with all required attributes
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        BU_edge_index=BU_edge_index,
        root_index=root_index
    )
    
    data, slices = split(data, node_graph_id)
    return data, slices

class DropEdge:
    def __init__(self, tddroprate, budroprate):
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __call__(self, data):
        # Store original edge_index
        edge_index = data.edge_index

        # Handle top-down edges
        if self.tddroprate > 0:
            row = list(edge_index[0])
            col = list(edge_index[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = torch.LongTensor([row, col])
        else:
            new_edgeindex = edge_index

        # Handle bottom-up edges
        burow = list(edge_index[1])
        bucol = list(edge_index[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = torch.LongTensor([row, col])
        else:
            bunew_edgeindex = torch.LongTensor([burow, bucol])

        # Set new edge indices
        data.edge_index = new_edgeindex
        data.BU_edge_index = bunew_edgeindex

        # Ensure root index is set for each graph in the batch
        if not hasattr(data, 'batch'):
            # Single graph case
            data.root_index = torch.LongTensor([0])
            if hasattr(data, 'x'):
                data.root = data.x[0]
        else:
            # Batch case
            batch_size = data.batch.max().item() + 1
            root_indices = []
            roots = []
            for i in range(batch_size):
                batch_mask = data.batch == i
                graph_nodes = torch.nonzero(batch_mask).squeeze()
                root_idx = graph_nodes[0]
                root_indices.append(root_idx)
                if hasattr(data, 'x'):
                    roots.append(data.x[root_idx])
            
            data.root_index = torch.LongTensor(root_indices)
            if hasattr(data, 'x'):
                data.root = torch.stack(roots) if roots else None

        return data

import os.path as osp
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array
import random
import numpy as np
import scipy.sparse as sp

import os.path as osp
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array
import random
import numpy as np
import scipy.sparse as sp

class FNNDataset(InMemoryDataset):
    def __init__(self, root, name, feature='spacy', empty=False, transform=None, pre_transform=None, pre_filter=None):
        if name is None:
            raise ValueError("Dataset name cannot be None")
            
        self.name = name
        self.feature = feature
        self.root = root
        
        # Initialize private attributes for storing features and classes
        self._num_features = None
        self._num_classes = None
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        if not empty:
            try:
                loaded_data = torch.load(self.processed_paths[0])
                if isinstance(loaded_data, tuple) and len(loaded_data) == 5:
                    self.data, self.slices, self.train_idx, self.val_idx, self.test_idx = loaded_data
                    
                    # Calculate features and classes after loading data
                    self._num_features = self.data.x.size(1)
                    self._num_classes = len(torch.unique(self.data.y))
                else:
                    raise ValueError("Unexpected data format")
            except Exception as e:
                print(f"Error loading dataset: {str(e)}")
                print(f"Attempted to load from: {self.processed_paths[0]}")
                raise

    @property
    def num_features(self):
        """Get the number of features in the dataset"""
        if self._num_features is None and hasattr(self, 'data'):
            self._num_features = self.data.x.size(1)
        return self._num_features

    @property
    def num_classes(self):
        """Get the number of classes in the dataset"""
        if self._num_classes is None and hasattr(self, 'data'):
            self._num_classes = len(torch.unique(self.data.y))
        return self._num_classes

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw/')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed/')

    @property
    def raw_file_names(self):
        return ['node_graph_id.npy', 'graph_labels.npy']

    @property
    def processed_file_names(self):
        base_name = f'{self.name[:3]}_data_{self.feature}'
        return [f'{base_name}_prefiler.pt'] if self.pre_filter else [f'{base_name}.pt']

    def download(self):
        raise NotImplementedError('Please download the data manually')

    def process(self):
        # Load and process the data
        self.data, self.slices = read_graph_data(self.raw_dir, self.feature)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        # Load split indices
        self.train_idx = torch.from_numpy(np.load(self.raw_dir + 'train_idx.npy')).to(torch.long)
        self.val_idx = torch.from_numpy(np.load(self.raw_dir + 'val_idx.npy')).to(torch.long)
        self.test_idx = torch.from_numpy(np.load(self.raw_dir + 'test_idx.npy')).to(torch.long)

        # Calculate number of features and classes
        self._num_features = self.data.x.size(1)
        self._num_classes = len(torch.unique(self.data.y))

        # Save everything
        torch.save((self.data, self.slices, self.train_idx, self.val_idx, self.test_idx),
                self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
        
    def get_num_classes(self):
        """
        Returns the number of classes in the dataset
        """
        if not hasattr(self, 'num_classes'):
            self.num_classes = len(torch.unique(self.data.y))
        return self.num_classes

    def get_num_features(self):
        """
        Returns the number of features in the dataset
        """
        if not hasattr(self, 'num_features'):
            self.num_features = self.data.x.size(1)
        return self.num_features