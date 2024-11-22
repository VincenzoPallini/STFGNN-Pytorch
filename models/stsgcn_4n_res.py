# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbedding(nn.Module):
    def __init__(self, input_length, num_of_vertices, embedding_size,
                 temporal=True, spatial=True):
        super(PositionEmbedding, self).__init__()
        self.temporal = temporal
        self.spatial = spatial
        if self.temporal:
            # shape (1, T, 1, C)
            self.temporal_emb = nn.Parameter(torch.randn(1, input_length, 1, embedding_size))
        if self.spatial:
            # shape (1, 1, N, C)
            self.spatial_emb = nn.Parameter(torch.randn(1, 1, num_of_vertices, embedding_size))

    def forward(self, x):
        # x shape: (B, T, N, C)
        if self.temporal:
            x = x + self.temporal_emb
        if self.spatial:
            x = x + self.spatial_emb
        return x


class GCNOperation(nn.Module):
    def __init__(self, adj, num_of_features, num_of_filter, activation):
        super(GCNOperation, self).__init__()
        self.adj = adj  # adj is a tensor of shape (4N, 4N)
        self.activation_type = activation
        self.num_of_filter = num_of_filter
        if activation == 'GLU':
            self.fc = nn.Linear(num_of_features, 2 * num_of_filter)
        elif activation == 'relu':
            self.fc = nn.Linear(num_of_features, num_of_filter)
        else:
            raise ValueError("Unsupported activation type")

    def forward(self, x):
        # Perform adjacency matrix multiplication
        x = torch.einsum('ij,jbc->ibc', self.adj, x)  # (4N, B, C)
        x = x.permute(1, 0, 2)  # (B, 4N, C)
        x = self.fc(x)  # Apply linear transformation

        # Apply activation function
        if self.activation_type == 'GLU':
            lhs, rhs = torch.chunk(x, 2, dim=2)  # Split into two parts
            x = lhs * torch.sigmoid(rhs)
        elif self.activation_type == 'relu':
            x = F.relu(x)

        x = x.permute(1, 0, 2)  # (4N, B, C')
        return x


class STSGCM(nn.Module):
    def __init__(self, adj, num_of_features, filters, activation):
        super(STSGCM, self).__init__()
        self.adj = adj
        self.num_layers = len(filters)
        self.gcn_layers = nn.ModuleList()
        self.filters = filters
        self.activation = activation
        for i in range(self.num_layers):
            gcn_layer = GCNOperation(
                adj=adj,
                num_of_features=num_of_features,
                num_of_filter=filters[i],
                activation=activation
            )
            self.gcn_layers.append(gcn_layer)
            num_of_features = filters[i]  # Update for next layer

    def forward(self, x):
        # x shape: (4N, B, C)
        need_concat = []
        num_of_vertices = x.shape[0] // 4
        for i, gcn in enumerate(self.gcn_layers):
            x = gcn(x)
            # Slice and expand dimensions
            sliced_x = x[num_of_vertices:2*num_of_vertices]  # (N, B, C')
            sliced_x = sliced_x.unsqueeze(0)  # (1, N, B, C')
            need_concat.append(sliced_x)
        # Concatenate over the new dimension
        need_concat = torch.cat(need_concat, dim=0)  # (num_layers, N, B, C')
        # Max over the concatenated dimension
        x = torch.max(need_concat, dim=0)[0]  # (N, B, C')
        return x


class STSGCL(nn.Module):
    def __init__(self, input_length, num_of_vertices, num_of_features, filters,
                 adj, activation='GLU', temporal_emb=True, spatial_emb=True):
        super(STSGCL, self).__init__()
        self.input_length = input_length
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.activation = activation
        self.adj = adj
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        self.position_embedding = PositionEmbedding(
            input_length=input_length,
            num_of_vertices=num_of_vertices,
            embedding_size=num_of_features,
            temporal=temporal_emb,
            spatial=spatial_emb
        )

        # Time convolution layers
        self.time_conv_left = nn.Conv2d(num_of_features, num_of_features, kernel_size=(1, 2), dilation=(1, 3))
        self.time_conv_right = nn.Conv2d(num_of_features, num_of_features, kernel_size=(1, 2), dilation=(1, 3))

        # STSGCM modules for each time step
        self.num_time_steps = input_length - 3
        self.stsgcm_layers = nn.ModuleList()
        for _ in range(self.num_time_steps):
            stsgcm = STSGCM(adj=adj, num_of_features=num_of_features, filters=filters, activation=activation)
            self.stsgcm_layers.append(stsgcm)

    def forward(self, x):
        # x shape: (B, T, N, C)
        x = self.position_embedding(x)
        # Transpose to (B, C, N, T)
        x_temp = x.permute(0, 3, 2, 1)
        # Apply time convolution
        data_left = torch.sigmoid(self.time_conv_left(x_temp))
        data_right = torch.tanh(self.time_conv_right(x_temp))
        data_time_axis = data_left * data_right  # (B, C, N, T-3)
        # Transpose back to (B, T-3, N, C)
        data_res = data_time_axis.permute(0, 3, 2, 1)  # (B, T-3, N, C)

        need_concat = []
        for i in range(self.num_time_steps):
            # Slice x[:, i:i+4, :, :]
            t = x[:, i:i+4, :, :]  # (B, 4, N, C)
            # Reshape to (B, 4N, C)
            t = t.reshape(-1, 4 * self.num_of_vertices, self.num_of_features)
            # Transpose to (4N, B, C)
            t = t.permute(1, 0, 2)
            # Apply STSGCM
            t = self.stsgcm_layers[i](t)  # (N, B, C')
            # Transpose to (B, N, C')
            t = t.permute(1, 0, 2)
            # Expand dims to (B, 1, N, C')
            t = t.unsqueeze(1)
            need_concat.append(t)
        # Concatenate over time dimension
        need_concat_ = torch.cat(need_concat, dim=1)  # (B, T-3, N, C')
        # Adjust data_res to match the channel dimensions if necessary
        if data_res.shape[-1] != need_concat_.shape[-1]:
            data_res = data_res.permute(0, 1, 3, 2)  # (B, T-3, C, N)
            data_res = nn.Conv2d(self.num_of_features, self.filters[-1], kernel_size=1)(data_res)
            data_res = data_res.permute(0, 1, 3, 2)  # (B, T-3, N, C')
        # Add residual connection
        layer_out = need_concat_ + data_res  # (B, T-3, N, C')
        return layer_out


class OutputLayer(nn.Module):
    def __init__(self, num_of_vertices, input_length, num_of_features,
                 num_of_filters=128, predict_length=12):
        super(OutputLayer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.num_of_filters = num_of_filters
        self.predict_length = predict_length

        self.fc1 = nn.Linear(input_length * num_of_features, num_of_filters)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_of_filters, predict_length)

    def forward(self, x):
        # x shape: (B, T, N, C)
        x = x.permute(0, 2, 1, 3)  # (B, N, T, C)
        x = x.reshape(x.shape[0], self.num_of_vertices, -1)  # (B, N, T*C)
        x = self.fc1(x)  # (B, N, num_of_filters)
        x = self.relu(x)
        x = self.fc2(x)  # (B, N, predict_length)
        x = x.permute(0, 2, 1)  # (B, predict_length, N)
        return x


class STSGCN(nn.Module):
    def __init__(self, num_nodes, input_dim, horizon, num_layers, filters,
                 module_type, adj, activation='GLU', use_mask=True, temporal_emb=True,
                 spatial_emb=True, first_layer_embedding_size=None, predict_length=12, rho=1):
        super(STSGCN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.filters = filters
        self.module_type = module_type
        self.activation = activation
        self.use_mask = use_mask
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.predict_length = predict_length
        self.rho = rho
        self.first_layer_embedding_size = first_layer_embedding_size

        if use_mask:
            self.mask = nn.Parameter(torch.ones_like(adj))
            self.adj = nn.Parameter(adj * self.mask)
        else:
            self.adj = adj

        # If first_layer_embedding_size is specified, add a linear layer
        if first_layer_embedding_size and first_layer_embedding_size != input_dim:
            self.first_layer_embedding = nn.Sequential(
                nn.Linear(input_dim, first_layer_embedding_size),
                nn.ReLU()
            )
            num_of_features = first_layer_embedding_size
        else:
            self.first_layer_embedding = None
            num_of_features = input_dim

        self.layers = nn.ModuleList()
        input_length = horizon
        for idx in range(num_layers):
            stsgcl = STSGCL(
                input_length=input_length,
                num_of_vertices=num_nodes,
                num_of_features=num_of_features,
                filters=filters[idx],
                adj=self.adj,
                activation=activation,
                temporal_emb=temporal_emb,
                spatial_emb=spatial_emb
            )
            self.layers.append(stsgcl)
            input_length -= 3
            num_of_features = filters[idx][-1]

        self.output_layer = OutputLayer(
            num_of_vertices=num_nodes,
            input_length=input_length,
            num_of_features=num_of_features,
            predict_length=predict_length
        )

    def forward(self, x, labels=None):
        # x shape: (B, T, N, C)
        if self.first_layer_embedding is not None:
            # Apply the first layer embedding
            B, T, N, C = x.shape
            x = x.view(-1, C)  # (B*T*N, C)
            x = self.first_layer_embedding(x)  # (B*T*N, first_layer_embedding_size)
            x = x.view(B, T, N, -1)  # (B, T, N, first_layer_embedding_size)

        for layer in self.layers:
            x = layer(x)
        # Output shape: (B, predict_length, N)
        output = self.output_layer(x)
        if labels is not None:
            loss = self.huber_loss(output, labels, rho=self.rho)
            return loss, output
        else:
            return output

    @staticmethod
    def huber_loss(pred, target, rho=1):
        abs_diff = torch.abs(pred - target)
        quadratic = torch.min(abs_diff, torch.tensor(rho))
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + rho * linear
        return loss.mean()
