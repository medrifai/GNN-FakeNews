import os
import os.path as osp
import traceback
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.io import read_txt_array
import numpy as np
import scipy.sparse as sp

class FNNDataset(InMemoryDataset):
    def __init__(self, root, feature, name="politifact", transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.feature = feature
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load_data()

    @property
    def raw_file_names(self):
        return [f'new_profile_feature.npz', 'A.txt', 'node_graph_id.npy', 'graph_labels.npy']

    @property
    def processed_file_names(self):
        return [f'{self.name}_{self.feature}_data.pt']

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name.lower(), 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name.lower(), 'processed')
    
    

    def load_data(self):
        processed_path = self.processed_paths[0]
        print(f"Trying to load processed data from: {processed_path}")
        print(f"Processed file exists: {osp.exists(processed_path)}")
        if osp.exists(processed_path):
            try:
                self.data, self.slices = torch.load(processed_path)
                print(f"Successfully loaded processed data with {len(self)} samples")
            except Exception as e:
                print(f"Error loading processed data: {e}. Reprocessing...")
                self.process()
        else:
            print("Processed data not found. Processing raw data...")
            self.process()

    def process(self):
        try:
            # Load node features
            feature_path = osp.join(self.raw_dir, f'new_{self.feature}_feature.npz')
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"Feature file not found at {feature_path}")
            print(f"Loading features from: {feature_path}")
            features = sp.load_npz(feature_path)
            x = torch.from_numpy(features.todense()).float()
            print(f"Loaded features with shape: {x.shape}")

            # Load adjacency matrix
            edge_path = osp.join(self.raw_dir, 'A.txt')
            if not os.path.exists(edge_path):
                raise FileNotFoundError(f"Edge file not found at {edge_path}")
            print(f"Loading edges from: {edge_path}")
            edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t().contiguous()
            print(f"Loaded edge index with shape: {edge_index.shape}")

            # Load node_graph_id and graph_labels
            node_graph_id_path = osp.join(self.raw_dir, 'node_graph_id.npy')
            graph_labels_path = osp.join(self.raw_dir, 'graph_labels.npy')
            
            if not os.path.exists(node_graph_id_path):
                raise FileNotFoundError(f"node_graph_id file not found at {node_graph_id_path}")
            if not os.path.exists(graph_labels_path):
                raise FileNotFoundError(f"graph_labels file not found at {graph_labels_path}")
                
            print(f"Loading node_graph_id from: {node_graph_id_path}")
            node_graph_id = torch.from_numpy(np.load(node_graph_id_path))
            print(f"Loading graph_labels from: {graph_labels_path}")
            graph_labels = torch.from_numpy(np.load(graph_labels_path)).long()

            print(f"Loaded node_graph_id with shape: {node_graph_id.shape}")
            print(f"Loaded graph_labels with shape: {graph_labels.shape}")

            # Create data list for each graph
            data_list = []
            
            # Process each graph separately
            unique_graph_ids = torch.unique(node_graph_id)
            print(f"Found {len(unique_graph_ids)} unique graphs")
            
            for graph_id in unique_graph_ids:
                graph_mask = node_graph_id == graph_id
                graph_nodes = graph_mask.nonzero(as_tuple=True)[0]
                
                # Get subgraph data
                graph_x = x[graph_nodes]
                graph_y = graph_labels[graph_id].long()
                
                # Get subgraph edges
                edge_mask = torch.isin(edge_index[0], graph_nodes) & torch.isin(edge_index[1], graph_nodes)
                graph_edges = edge_index[:, edge_mask]
                
                # Remap node indices
                node_idx_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(graph_nodes)}
                graph_edges = torch.tensor([[node_idx_map[edge[0].item()], node_idx_map[edge[1].item()]] 
                                        for edge in graph_edges.t()]).t()
                
                # Create Data object for this graph
                data = Data(
                    x=graph_x,
                    edge_index=graph_edges,
                    y=graph_y,
                    root_index=torch.tensor([0],dtype=torch.long)  # Root is always the first node
                )
                
                data_list.append(data)
            
            print(f"Created {len(data_list)} graph objects")
            
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            
            # Set attributes
            self.data = data
            self.slices = slices
            
            print(f"Successfully processed and saved dataset with {len(data_list)} graphs")

        except Exception as e:
            print(f"Error processing data: {e}")
            traceback.print_exc()
            raise

    def get(self, idx):
        if not hasattr(self, "data") or not hasattr(self, "slices"):
            self.load_data()  # Ensure data is loaded if not already.
        return super().get(idx)


