from experiments.models.neural_net import PytorchModel
import tenseal as ts
import torch
import numpy as np

class PrivatePytorchModel(PytorchModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encrypted_context = None

    def set_encryption_context(self, context):
        self.encrypted_context = context

    def private_forward(self, enc_x):
        """
        Generalized encrypted forward pass for arbitrary model depth.
        Uses polynomial activations for all non-linearities.
        """
        enc_out = enc_x
        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                w = layer.weight.data.cpu().numpy()
                b = layer.bias.data.cpu().numpy()
                if w.shape[0] == 1:
                    enc_out = enc_out.dot(w[0].tolist()) + b[0]
                else:
                    enc_out = [enc_out.dot(wi.tolist()) + bi for wi, bi in zip(w, b)]
            elif isinstance(layer, (torch.nn.ReLU, torch.nn.Sigmoid)):
                # Use a cubic polynomial approximation for all activations
                # Example: sigmoid ≈ 0.5 + 0.197x - 0.004x^3
                if isinstance(enc_out, list):
                    enc_out = [eo.polyval([0.5, 0.197, 0, -0.004]) for eo in enc_out]
                else:
                    enc_out = enc_out.polyval([0.5, 0.197, 0, -0.004])
            elif isinstance(layer, torch.nn.Dropout):
                continue  # Dropout is ignored in encrypted inference
            else:
                raise NotImplementedError(f"Layer {type(layer)} not supported in encrypted forward.")
        return enc_out

    def private_backward(self, enc_x, enc_y, enc_out, learning_rate=0.01):
        """
        Toy encrypted backward pass for a single linear layer.
        This is NOT secure or practical for real use, but demonstrates the concept.
        Steps:
        1. Decrypt error and input (insecure, for demo only).
        2. Compute gradients in plaintext.
        3. Update weights and bias in plaintext.
        """

        # Decrypt (for demonstration only)
        x = enc_x.decrypt()
        y = enc_y.decrypt()
        out = enc_out.decrypt()
        # Compute error
        error = out - y
        # Compute gradients
        grad_w = [xi * error for xi in x]
        grad_b = error
        # Update weights and bias (for first linear layer)
        linear = self.model[0]
        w = linear.weight.data.cpu().numpy()
        b = linear.bias.data.cpu().numpy()
        # Assume single output neuron for simplicity
        w[0] -= learning_rate * np.array(grad_w)
        b[0] -= learning_rate * grad_b
        linear.weight.data = torch.tensor(w, dtype=linear.weight.data.dtype)
        linear.bias.data = torch.tensor(b, dtype=linear.bias.data.dtype)

# Example usage:
# context = ts.context(...)
# model = PrivatePytorchModel(...)
# model.set_encryption_context(context)
# enc_x = ts.ckks_vector(context, x.tolist())
# enc_out = model.private_forward(enc_x)
