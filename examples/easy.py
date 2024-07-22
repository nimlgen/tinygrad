from tinygrad import Tensor, dtypes
print(Tensor([25.0], dtype=dtypes.float64).sin().numpy())