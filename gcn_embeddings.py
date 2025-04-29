import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Step 1: Prepare the interaction matrix
# Example: 3 users, 4 items

def gcd_embeddings(matrix, epochs=50):
    # Use the actual input matrix instead of the example one
    R = torch.tensor(matrix, dtype=torch.float)

    # Step 2: Construct the bipartite graph adjacency matrix
    num_users, num_items = R.shape
    A = torch.zeros((num_users + num_items, num_users + num_items))
    A[:num_users, num_users:] = R
    A[num_users:, :num_users] = R.T

    # Step 3: Normalize the adjacency matrix
    D = torch.diag(A.sum(1))
    # Handle zero degrees by adding a small value to avoid division by zero
    D_diag = torch.diag(D)
    D_diag[D_diag == 0] = 1e-6  # Add small value to zeros
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_diag))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    # Step 4: Define the GCN model
    class GCN(torch.nn.Module):
        def __init__(self, num_nodes, embedding_dim):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(num_nodes, embedding_dim)
            self.conv2 = GCNConv(embedding_dim, embedding_dim)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x

    # Step 5: Prepare data for PyTorch Geometric
    edge_index = A.nonzero().t()  # Convert adjacency matrix to edge index format
    user_item_edges = edge_index[:, edge_index[0] < num_users]
    x = torch.eye(num_users + num_items)  # Identity matrix as initial node features

    data = Data(x=x, edge_index=user_item_edges)

    # Step 6: Train the GCN
    model = GCN(num_users + num_items, embedding_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # List to store loss values for plotting
    loss_values = []

    for epoch in tqdm(range(epochs), desc="Training GCN", ncols=100):
        model.train()
        optimizer.zero_grad()
        
        embeddings = model(data.x, data.edge_index)
        user_embeddings = embeddings[:num_users]
        item_embeddings = embeddings[num_users:]
        
        # Skip the batch if there are no user-item edges
        if user_item_edges.shape[1] == 0:
            print("Warning: No user-item edges found. Cannot compute loss.")
            break
            
        true_ratings = R[user_item_edges[0], user_item_edges[1] - num_users]
        pred_ratings = (user_embeddings[user_item_edges[0]] * item_embeddings[user_item_edges[1] - num_users]).sum(dim=1)
        loss = F.mse_loss(pred_ratings, true_ratings)
        
        # Save loss value for plotting
        loss_values.append(loss.item())

        if (epoch + 1) % 20 == 0:
            tqdm.write(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.10f}')
            
        loss.backward()
        optimizer.step()

    # Step 7: Extract embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        user_embeddings = embeddings[:num_users]
        item_embeddings = embeddings[num_users:]

    # Plot the loss changes over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', linestyle='-', markersize=2)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('gcn_training_loss.png')
    plt.close()
    
    print(f"Training complete. Loss plot saved as 'gcn_training_loss.png'")
    print(f"Final loss: {loss_values[-1]:.6f}")

    return user_embeddings, item_embeddings

if __name__ == "__main__":
    matrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0]
    ])
    gcd_embeddings(matrix)