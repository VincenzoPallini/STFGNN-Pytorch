# -*- coding:utf-8 -*-

import time
import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils_4n0_3layer_12T_res import (
    construct_model,
    generate_data,
    masked_mae_np,
    masked_mape_np,
    masked_mse_np
)

# Import additional required libraries
from torch.optim.lr_scheduler import LambdaLR

# Function to save and show plots
def save_and_show_plot(fig, filename):
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Plot saved to {filepath}")

# Function to analyze and visualize errors
def analyze_errors(predictions, true_values):
    errors = np.abs(predictions - true_values)
    
    # 1. Plot the distribution of errors
    fig1, ax1 = plt.subplots()
    ax1.hist(errors.flatten(), bins=50, color='gray', alpha=0.7)
    ax1.set_title('Distribution of Absolute Errors')
    ax1.set_xlabel('Absolute Error')
    ax1.set_ylabel('Frequency')
    save_and_show_plot(fig1, 'distribution_of_errors.png')

    # 2. Find points with maximum and minimum error
    max_error_idx = np.argmax(errors.flatten())
    min_error_idx = np.argmin(errors.flatten())

    print(f'Maximum error: {errors.flatten()[max_error_idx]:.2f} at point {max_error_idx}')
    print(f'Minimum error: {errors.flatten()[min_error_idx]:.2f} at point {min_error_idx}')

    # Plot predictions vs true values highlighting the max and min errors
    fig2, ax2 = plt.subplots()
    ax2.plot(true_values.flatten(), label="True Values", color='blue', alpha=0.6)
    ax2.plot(predictions.flatten(), label="Predictions", color='orange', alpha=0.6)

    # Highlight points with maximum and minimum error
    ax2.scatter(max_error_idx, predictions.flatten()[max_error_idx], color='red', label="Max Error", s=100)
    ax2.scatter(min_error_idx, predictions.flatten()[min_error_idx], color='green', label="Min Error", s=100)

    ax2.set_title("Predictions vs True Values with Max and Min Errors")
    ax2.set_xlabel("Observation Point")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True)
    save_and_show_plot(fig2, 'predictions_vs_true_values.png')

    # 3. Plot the error for each observation point
    fig3, ax3 = plt.subplots()
    ax3.plot(errors.flatten(), color='red')
    ax3.set_title('Absolute Error per Observation Point')
    ax3.set_xlabel('Observation Point')
    ax3.set_ylabel('Absolute Error')
    ax3.grid(True)
    save_and_show_plot(fig3, 'error_per_observation_point.png')

    # 4. Statistical analysis of errors
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    print(f"Mean error: {mean_error:.2f}")
    print(f"Standard deviation of errors: {std_error:.2f}")
    print(f"Maximum error: {max_error:.2f}")
    print(f"Minimum error: {min_error:.2f}")

# Function to calculate mean absolute errors (MAE)
def calculate_errors(predictions, true_values):
    errors = np.abs(predictions - true_values)
    mae_per_node = np.mean(errors, axis=1)
    return mae_per_node

# Function to calculate graph structural features
def calculate_graph_features(G):
    degree_centrality = nx.degree_centrality(G)
    densities = {}
    for node in G.nodes():
        ego = nx.ego_graph(G, node, radius=2)
        densities[node] = nx.density(ego)
    return degree_centrality, densities

# Function to visualize the mean error over time for selected nodes
def plot_mean_error_for_nodes(mae_per_node, nodes, title, output_dir, filename):
    plt.figure(figsize=(10, 5))
    for node in nodes:
        plt.plot(mae_per_node[:, node], label=f'Node {node}')
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"{title} saved to: {os.path.join(output_dir, filename)}")

# Function to perform statistical analysis
def statistical_analysis(degree_centrality, densities, mae_per_node, output_dir):
    nodes = list(degree_centrality.keys())
    degree_values = [degree_centrality[node] for node in nodes]
    density_values = [densities[node] for node in nodes]
    mean_errors = [np.mean(mae_per_node[:, node]) for node in nodes]

    # Pearson correlation between degree centrality and mean error
    corr_degree, p_value_degree = pearsonr(degree_values, mean_errors)
    print(f"Correlation between Degree Centrality and Mean Error: {corr_degree:.4f} (p-value: {p_value_degree:.4e})")

    # Pearson correlation between subgraph density and mean error
    corr_density, p_value_density = pearsonr(density_values, mean_errors)
    print(f"Correlation between Subgraph Density and Mean Error: {corr_density:.4f} (p-value: {p_value_density:.4e})")

    # Linear regression between degree centrality and mean error
    degree_values_np = np.array(degree_values).reshape(-1, 1)
    mean_errors_np = np.array(mean_errors)
    reg_degree = LinearRegression().fit(degree_values_np, mean_errors_np)
    print(f"Linear Regression (Degree Centrality vs Mean Error): Coefficient={reg_degree.coef_[0]:.4f}, Intercept={reg_degree.intercept_:.4f}")

    # Linear regression between subgraph density and mean error
    density_values_np = np.array(density_values).reshape(-1, 1)
    reg_density = LinearRegression().fit(density_values_np, mean_errors_np)
    print(f"Linear Regression (Subgraph Density vs Mean Error): Coefficient={reg_density.coef_[0]:.4f}, Intercept={reg_density.intercept_:.4f}")

    # Scatter plot for degree centrality vs mean error
    plt.figure(figsize=(10, 5))
    plt.scatter(degree_values, mean_errors, color='b', alpha=0.6)
    plt.plot(degree_values, reg_degree.predict(degree_values_np), color='r', linewidth=2)
    plt.xlabel("Degree Centrality")
    plt.ylabel("Mean Error")
    plt.title("Degree Centrality vs Mean Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'degree_centrality_vs_mean_error.png'))
    plt.close()

    # Scatter plot for subgraph density vs mean error
    plt.figure(figsize=(10, 5))
    plt.scatter(density_values, mean_errors, color='b', alpha=0.6)
    plt.plot(density_values, reg_density.predict(density_values_np), color='r', linewidth=2)
    plt.xlabel("Subgraph Density")
    plt.ylabel("Mean Error")
    plt.title("Subgraph Density vs Mean Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subgraph_density_vs_mean_error.png'))
    plt.close()

# Function to visualize node predictions
def visualize_node_predictions(predictions, true_values, adj_mx, node_id, k=2, j=3, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G = nx.from_numpy_array(adj_mx)
    subgraph = nx.ego_graph(G, node_id, radius=k)
    subgraph_nodes = list(subgraph.nodes())

    sub_predictions = predictions[:, subgraph_nodes, :]
    sub_true_values = true_values[:, subgraph_nodes, :]
    errors = np.abs(sub_predictions - sub_true_values)

    t_pred = min(predictions.shape[2] - 1, j - 1)
    times = list(range(max(0, t_pred - j + 1), t_pred + 1))

    fig, axes = plt.subplots(len(times), 3, figsize=(20, 7 * len(times)))
    fig.suptitle(f"Analysis of Node {node_id} and Its Subgraph (distance {k})", fontsize=16, y=1.02)

    txt_filename = os.path.join(output_dir, f"node_{node_id}_values.txt")
    with open(txt_filename, 'w') as f:
        f.write(f"Numerical values for node {node_id} and its subgraph:\n\n")

    pos = nx.spring_layout(subgraph)
    label_offset_x = 0
    label_offset_y = 0.1
    central_node_color = 'yellow'
    central_node_size = 500

    for i, t in enumerate(times):
        for ax_idx, (data, title) in enumerate([(sub_true_values, "Ground Truth"), 
                                                (sub_predictions, "Prediction"), 
                                                (errors, "Absolute Error")]):
            ax = axes[i, ax_idx]
            node_color_list = [central_node_color if node == node_id else 'lightblue' for node in subgraph_nodes]
            node_size_list = [central_node_size if node == node_id else 300 for node in subgraph_nodes]

            nx.draw(subgraph, pos, ax=ax, node_color=node_color_list, node_size=node_size_list,
                    cmap='viridis' if ax_idx < 2 else 'Reds', with_labels=False)

            node_ids = {node: str(node) for node in subgraph_nodes}
            nx.draw_networkx_labels(subgraph, pos, labels=node_ids, ax=ax, font_size=8)

            labels_values = {node: f"{data[0, n, t]:.2f}" for n, node in enumerate(subgraph_nodes)}
            label_pos_values = {node: (pos[node][0] + label_offset_x, pos[node][1] + label_offset_y) for node in subgraph_nodes}

            font_color = 'blue' if title != "Absolute Error" else 'red'
            nx.draw_networkx_labels(subgraph, label_pos_values, labels_values, ax=ax, font_size=8, font_color=font_color)

            ax.set_title(f"{title} (t={t})", fontsize=12, pad=20)

            if ax_idx == 0 and i == 0:
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', label='Central Node', markerfacecolor=central_node_color, markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', label='Other Nodes', markerfacecolor='lightblue', markersize=8)
                ]
                ax.legend(handles=legend_elements, loc='upper right')

        with open(txt_filename, 'a') as f:
            f.write(f"Time t={t}:\n")
            for n, (true_val, pred_val, err_val) in enumerate(zip(sub_true_values[0, :, t],
                                                                  sub_predictions[0, :, t],
                                                                  errors[0, :, t])):
                f.write(f"  Node {subgraph_nodes[n]}: True={true_val:.4f}, Pred={pred_val:.4f}, Err={err_val:.4f}\n")
            f.write("\n")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f"node_{node_id}_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph saved as node_{node_id}_analysis.png in folder '{output_dir}'")
    print(f"Numerical values saved as node_{node_id}_values.txt in folder '{output_dir}'")

    # Calculate errors and graph structural features
    mae_per_node = calculate_errors(predictions, true_values)

    # Heatmap of errors
    plt.figure(figsize=(12, 8))
    sns.heatmap(mae_per_node.T, cmap='Reds', cbar_kws={'label': 'Mean Absolute Error (MAE)'})
    plt.title("Heatmap of Errors per Node and Time Step")
    plt.xlabel("Time Steps")
    plt.ylabel("Nodes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_of_errors.png'))
    plt.close()
    print(f"Heatmap of errors saved in: {os.path.join(output_dir, 'heatmap_of_errors.png')}")

# Function to plot sample outputs
def plot_sample_output(outputs, labels):
    print(f"Shape of outputs: {outputs.shape}, Shape of labels: {labels.shape}")
    
    if outputs.shape[0] < 1 or labels.shape[0] < 1:
        print("Not enough data to create the plot.")
        return
    
    sample_output = outputs[0]
    sample_label = labels[0]
    
    print(f"Shape of sample_output: {sample_output.shape}, Shape of sample_label: {sample_label.shape}")
    
    if sample_output.shape != sample_label.shape:
        print("Sample dimensions do not match between output and labels.")
        return
    
    flat_output = sample_output.flatten()
    flat_label = sample_label.flatten()
    
    errors = np.abs(flat_output - flat_label)
    
    max_error_idx = np.argmax(errors)
    min_error_idx = np.argmin(errors)
    
    print(f"Max error index: {max_error_idx}, Min error index: {min_error_idx}")
    print(f"Maximum error: {errors[max_error_idx]}, Minimum error: {errors[min_error_idx]}")

    # Plot for the entire sample
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(flat_label, color='blue', label='True Values', alpha=0.7)
    ax1.plot(flat_output, color='orange', label='Predictions', alpha=0.7)
    
    # Highlight max and min error points
    ax1.scatter(max_error_idx, flat_output[max_error_idx], color='red', label='Max Error')
    ax1.scatter(min_error_idx, flat_output[min_error_idx], color='green', label='Min Error')
    ax1.set_title('Predictions vs True Values (Sample)')
    ax1.set_xlabel('Observation Point')
    ax1.set_ylabel('Value')
    ax1.legend()
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'full_sample_comparison.png'))
    plt.close(fig1)

    # Zoomed plot
    zoom_range = min(500, len(flat_output))
    fig2, ax2 = plt.subplots(figsize=(15, 6))
    ax2.plot(range(zoom_range), flat_label[:zoom_range], color='blue', label='True Values', alpha=0.7)
    ax2.plot(range(zoom_range), flat_output[:zoom_range], color='orange', label='Predictions', alpha=0.7)
    
    zoom_errors = errors[:zoom_range]
    max_error_idx_zoom = np.argmax(zoom_errors)
    min_error_idx_zoom = np.argmin(zoom_errors)
    
    ax2.scatter(max_error_idx_zoom, flat_output[max_error_idx_zoom], color='red', label='Max Error (zoom)')
    ax2.scatter(min_error_idx_zoom, flat_output[min_error_idx_zoom], color='green', label='Min Error (zoom)')
    ax2.set_title(f'Zoom on the first {zoom_range} points of the sample')
    ax2.set_xlabel('Observation Point')
    ax2.set_ylabel('Value')
    ax2.legend()
    
    plt.savefig(os.path.join(output_dir, f'zoom_{zoom_range}_sample_comparison.png'))
    plt.close(fig2)

    print("All plots have been saved in the 'output' folder.")

# Main code
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--test", action="store_true", help="test program")
parser.add_argument("--plot", help="plot network graph", action="store_true")
parser.add_argument("--save", action="store_true", help="save model")
args = parser.parse_args()

config_filename = args.config

with open(config_filename, 'r') as f:
    config = json.loads(f.read())

print(json.dumps(config, sort_keys=True, indent=4))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = construct_model(config).to(device)

batch_size = config['batch_size']
num_of_vertices = config['num_of_vertices']
graph_signal_matrix_filename = config['graph_signal_matrix_filename']

loaders = []
true_values = []
for idx, (x, y) in enumerate(generate_data(graph_signal_matrix_filename)):
    if args.test:
        x = x[: 100]
        y = y[: 100]
    y = y.squeeze(axis=-1)
    print(x.shape, y.shape)
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    shuffle = (idx == 0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    loaders.append(loader)
    if idx == 0:
        training_samples = x.shape[0]
    else:
        true_values.append(y)

train_loader, val_loader, test_loader = loaders
val_y, test_y = true_values
val_y = torch.tensor(val_y, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)

epochs = config['epochs']

optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'])
criterion = nn.MSELoss()

def lr_lambda(epoch):
    return (1 - epoch / epochs) ** 2

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

num_of_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Number of Parameters: {}".format(num_of_parameters), flush=True)

def training(epochs):
    lowest_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        t = time.time()
        net.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f'Training: Epoch: {epoch}, Loss: {train_loss:.4f}, Time: {time.time() - t:.2f}s')
        
        net.eval()
        with torch.no_grad():
            val_predictions = []
            for data, target in val_loader:
                data = data.to(device)
                output = net(data)
                val_predictions.append(output.cpu().numpy())
            val_predictions = np.concatenate(val_predictions, axis=0)
            val_loss = masked_mae_np(val_y.numpy(), val_predictions, 0)
        print(f'Validation: Epoch: {epoch}, Loss: {val_loss:.4f}, Time: {time.time() - t:.2f}s')

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            with torch.no_grad():
                test_predictions = []
                for data, target in test_loader:
                    data = data.to(device)
                    output = net(data)
                    test_predictions.append(output.cpu().numpy())
                test_predictions = np.concatenate(test_predictions, axis=0)
                tmp_info = []
                for idx in range(config['num_for_predict']):
                    y = test_y[:, : idx + 1, :]
                    x = test_predictions[:, : idx + 1, :]
                    tmp_info.append((
                        masked_mae_np(y.numpy(), x, 0),
                        masked_mape_np(y.numpy(), x, 0),
                        masked_mse_np(y.numpy(), x, 0) ** 0.5
                    ))
                mae, mape, rmse = tmp_info[-1]
            print('Test: Epoch: {}, MAE: {:.2f}, MAPE: {:.2f}, RMSE: {:.2f}, '
                  'Time: {:.2f}s'.format(
                    epoch, mae, mape, rmse, time.time() - t))
            # Save the best model
            if args.save:
                torch.save(net.state_dict(), f'STSGCN_best.pth')
        scheduler.step()

if args.test:
    epochs = 5
training(epochs)

# Load the best model for evaluation
if args.save and os.path.exists('STSGCN_best.pth'):
    net.load_state_dict(torch.load('STSGCN_best.pth'))
else:
    print("Best model not saved, using the current model.")

net.eval()
with torch.no_grad():
    test_predictions = []
    for data, target in test_loader:
        data = data.to(device)
        output = net(data)
        test_predictions.append(output.cpu().numpy())
    predictions = np.concatenate(test_predictions, axis=0)
    true_values = test_y.numpy()

# Analyze errors and plot results
analyze_errors(predictions, true_values)
plot_sample_output(predictions, true_values)

# Load adjacency matrix
# Assuming the adjacency matrix is provided in the config file
adj_mx_filename = config.get('adj_mx_filename', None)
if adj_mx_filename and os.path.exists(adj_mx_filename):
    adj_mx = np.load(adj_mx_filename)
else:
    # If adjacency matrix is not provided, create a default one
    adj_mx = np.eye(num_of_vertices)
    print("Adjacency matrix not found, using identity matrix.")

# Visualize node predictions for selected nodes
for node_id in [0, 10, 20]:  # Example nodes
    visualize_node_predictions(predictions, true_values, adj_mx, node_id, j=3, output_dir='output')

# Calculate MAE per node and time step
mae_per_node = calculate_errors(predictions, true_values)

# Create the graph and calculate features
G = nx.from_numpy_array(adj_mx)
degree_centrality, densities = calculate_graph_features(G)

# Identify nodes with low and high degree centrality and density
sorted_degree_centrality = sorted(degree_centrality.items(), key=lambda x: x[1])
sorted_densities = sorted(densities.items(), key=lambda x: x[1])

# Get nodes with low and high degree centrality and density
low_degree_nodes = [node for node, _ in sorted_degree_centrality[:5]]
high_degree_nodes = [node for node, _ in sorted_degree_centrality[-5:]]
low_density_nodes = [node for node, _ in sorted_densities[:5]]
high_density_nodes = [node for node, _ in sorted_densities[-5:]]

# Plot mean error over time for selected nodes
plot_mean_error_for_nodes(mae_per_node, low_degree_nodes, "Mean Error for Nodes with Low Degree Centrality", output_dir='output', filename='low_degree_nodes_error.png')
plot_mean_error_for_nodes(mae_per_node, high_degree_nodes, "Mean Error for Nodes with High Degree Centrality", output_dir='output', filename='high_degree_nodes_error.png')
plot_mean_error_for_nodes(mae_per_node, low_density_nodes, "Mean Error for Nodes with Low Subgraph Density", output_dir='output', filename='low_density_nodes_error.png')
plot_mean_error_for_nodes(mae_per_node, high_density_nodes, "Mean Error for Nodes with High Subgraph Density", output_dir='output', filename='high_density_nodes_error.png')

# Perform statistical analysis
statistical_analysis(degree_centrality, densities, mae_per_node, output_dir='output')
