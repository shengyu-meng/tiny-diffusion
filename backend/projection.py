import torch

class RandomProjection:
    def __init__(self, input_dim: int, output_dim: int, seed: int = 42):
        if output_dim > input_dim:
            raise ValueError("Output dimension cannot be greater than input dimension.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed

        # Initialize a random projection matrix.
        # Using a Gaussian random matrix for simplicity and computational efficiency.
        # The matrix shape will be (input_dim, output_dim).
        torch.manual_seed(self.seed)
        self.projection_matrix = torch.randn(input_dim, output_dim)
        # Normalize columns for better numerical stability (often recommended).
        # This scales the columns so that their L2 norm is 1.
        self.projection_matrix = self.projection_matrix / torch.linalg.norm(self.projection_matrix, dim=0)

    def project(self, data: torch.Tensor) -> torch.Tensor:
        """
        Projects high-dimensional data to the lower-dimensional space.

        Args:
            data: A torch.Tensor of shape (..., input_dim).
                  The last dimension should be the feature dimension.
        Returns:
            A torch.Tensor of shape (..., output_dim).
                  The last dimension will be the projected dimension.
        """
        if data.shape[-1] != self.input_dim:
            raise ValueError(f"Last dimension of data ({data.shape[-1]}) must match input_dim ({self.input_dim}).")

        # Perform the matrix multiplication: data @ projection_matrix
        # torch.matmul handles broadcasting for leading dimensions.
        return torch.matmul(data, self.projection_matrix.to(data.device))

