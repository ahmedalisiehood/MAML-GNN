import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
from sklearn.decomposition import FastICA

class GAT_GCN_Model(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64, gat_heads=8):
        super(GAT_GCN_Model, self).__init__()
        self.gat = GATConv(in_channels, hidden_dim, heads=gat_heads, concat=True)
        self.node_embedding = nn.Linear(hidden_dim * gat_heads, hidden_dim)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat(x, edge_index)
        x = torch.nn.functional.elu(x)
        x = self.node_embedding(x)
        x = torch.nn.functional.elu(x)
        x = self.gcn1(x, edge_index)
        x = torch.nn.functional.elu(x)
        x = self.gcn2(x, edge_index)
        return x

class ICA_MMFL(nn.Module):
    def __init__(self, n_components):
        super(ICA_MMFL, self).__init__()
        self.ica = FastICA(n_components=n_components)

    def forward(self, ct_features, pet_features, fused_features):
        concatenated_features = torch.cat((ct_features, pet_features, fused_features), dim=1).cpu().detach().numpy()
        fused_output = self.ica.fit_transform(concatenated_features)
        fused_output = torch.tensor(fused_output).to(ct_features.device)
        return fused_output

class MAML_GNN_Model(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64, gat_heads=8, n_components=32):
        super(MAML_GNN_Model, self).__init__()
        self.ct_model = GAT_GCN_Model(in_channels, hidden_dim, hidden_dim, gat_heads)
        self.pet_model = GAT_GCN_Model(in_channels, hidden_dim, hidden_dim, gat_heads)
        self.fused_model = GAT_GCN_Model(in_channels, hidden_dim, hidden_dim, gat_heads)
        self.ica_mmfl = ICA_MMFL(n_components=n_components)
        self.classifier = nn.Linear(n_components, out_channels)

    def forward(self, ct_data, pet_data, fused_data):
        ct_output = self.ct_model(ct_data)
        pet_output = self.pet_model(pet_data)
        fused_output = self.fused_model(fused_data)
        fused_representation = self.ica_mmfl(ct_output, pet_output, fused_output)
        output = self.classifier(fused_representation)
        return torch.nn.functional.log_softmax(output, dim=1)
