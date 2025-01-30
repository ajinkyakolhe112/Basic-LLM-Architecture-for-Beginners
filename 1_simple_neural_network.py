import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork(nn.Module):
    """A simple 2-layer neural network for educational purposes.
    Architecture:
        Input Layer -> Hidden Layer -> ReLU -> Output Layer
    """
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the neural network.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in the hidden layer
            output_size (int): Number of output neurons
        """
        super().__init__()
        
        # First layer (input -> hidden)
        self.layer1     = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()  # Non-linear activation function
        
        # Second layer (hidden -> output)
        self.layer2     = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Pass through first layer and activation
        x = self.layer1(x)
        x = self.activation(x)
        
        # Pass through second layer
        x = self.layer2(x)
        return x

def generate_sample_data(n_samples=1000):
    """Generate synthetic data for training and testing.
    
    Returns:
        tuple: (X_train, y_train) where X_train is input features and y_train is labels
    """
    # Generate random input data
    X = np.random.randn(n_samples, 2)
    
    # Generate labels: 1 if x1 + x2 > 0, else 0 (simple decision boundary)
    y = np.array([1 if x1 + x2 > 0 else 0 for x1, x2 in X])
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    return X, y

def train_model(model, X_train, y_train, num_epochs=100, learning_rate=0.01):
    """Train the neural network.
    
    Args:
        model (SimpleNeuralNetwork): The neural network model
        X_train (torch.Tensor): Training features
        y_train (torch.Tensor): Training labels
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization
    
    Returns:
        list: Training losses over epochs
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Track losses
    losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss
        losses.append(loss.item())
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return losses

def plot_decision_boundary(model, X, y):
    """Plot the decision boundary of the trained model.
    
    Args:
        model (SimpleNeuralNetwork): Trained neural network model
        X (torch.Tensor): Input features
        y (torch.Tensor): True labels
    """
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Make predictions for each point in the mesh
    model.eval()
    with torch.no_grad():
        Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = torch.argmax(Z, dim=1).reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    X, y = generate_sample_data()
    
    # Create and train model
    model = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=2)
    losses = train_model(model, X, y)
    
    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    # Plot decision boundary
    plt.subplot(1, 2, 2)
    plot_decision_boundary(model, X, y)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
