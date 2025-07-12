import torch
import torch.nn as nn
from ultralytics.nn.modules import (
    C2f,
    Conv,
    Bottleneck,
    CIB,
    RepVGGDW,
)


def infer_shortcut(bottleneck):
    """
    Infer shortcut configuration and module type from a bottleneck module.
    
    This function analyzes a bottleneck module to determine:
    1. Whether it has a shortcut connection
    2. Whether it's a CIB (Cross-Information Bottleneck) module
    3. Whether it uses large kernel operations
    
    Args:
        bottleneck: A bottleneck module (either Bottleneck or CIB)
        
    Returns:
        tuple: (has_shortcut, is_CIB, has_large_kernel)
            - has_shortcut: Boolean indicating if shortcut connection exists
            - is_CIB: Boolean indicating if this is a CIB module
            - has_large_kernel: Boolean indicating if large kernel operations are used
    """
    if isinstance(bottleneck, Bottleneck):
        # For standard Bottleneck modules, check if input/output channels match
        # and if the module has an 'add' attribute (indicating shortcut)
        c1 = bottleneck.cv1.conv.in_channels
        c2 = bottleneck.cv2.conv.out_channels
        return c1 == c2 and hasattr(bottleneck, "add") and bottleneck.add, False, False
    else:
        # For CIB modules, check for large kernel operations in cv1
        check_large_kernel_option = [
            isinstance(mod, RepVGGDW) for mod in bottleneck.cv1
        ]
        return bottleneck.add, True, any(check_large_kernel_option)


class C2f_v2(nn.Module):
    """
    Enhanced CSP (Cross Stage Partial) Bottleneck with 2 convolutions.
    
    This is an improved version of the C2f module that separates the initial
    convolution into two separate paths (cv0 and cv1) instead of splitting
    the output of a single convolution. This design allows for better
    feature representation and more flexible architecture.
    
    Attributes:
        c (int): Number of hidden channels
        cv0 (Conv): First convolution layer for the main path
        cv1 (Conv): Second convolution layer for the shortcut path
        cv2 (Conv): Final convolution layer that combines all features
        m (nn.ModuleList): List of bottleneck modules (Bottleneck or CIB)
    """
    
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, is_CIB=False, lk=False
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        """
        Initialize C2f_v2 module.
        
        Args:
            c1 (int): Number of input channels
            c2 (int): Number of output channels
            n (int): Number of bottleneck modules
            shortcut (bool): Whether to use shortcut connections
            g (int): Number of groups for convolutions
            e (float): Expansion ratio for hidden channels
            is_CIB (bool): Whether to use CIB modules instead of Bottleneck
            lk (bool): Whether to use large kernel operations (for CIB)
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        
        # Two separate convolution paths instead of one split path
        self.cv0 = Conv(c1, self.c, 1, 1)  # Main path convolution
        self.cv1 = Conv(c1, self.c, 1, 1)  # Shortcut path convolution
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Final combination convolution
        
        # Create bottleneck modules based on type
        if not is_CIB:
            # Standard Bottleneck modules
            self.m = nn.ModuleList(
                Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                for _ in range(n)
            )
        else:
            # CIB (Cross-Information Bottleneck) modules
            self.m = nn.ModuleList(
                CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n)
            )

    def forward(self, x):
        """
        Forward pass through the C2f_v2 module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor with processed features
        """
        # Initialize feature list with outputs from both convolution paths
        y = [self.cv0(x), self.cv1(x)]
        
        # Process through bottleneck modules, each taking the last feature as input
        y.extend(m(y[-1]) for m in self.m)
        
        # Concatenate all features and pass through final convolution
        return self.cv2(torch.cat(y, 1))


def transfer_weights(c2f, c2f_v2):
    """
    Transfer weights from a C2f module to a C2f_v2 module.
    
    This function handles the weight transfer when converting from the original
    C2f architecture to the improved C2f_v2 architecture. It properly splits
    the weights from the single cv1 convolution to the two separate cv0 and cv1
    convolutions in C2f_v2.
    
    Args:
        c2f: Source C2f module
        c2f_v2: Target C2f_v2 module
    """
    # Transfer the final convolution and bottleneck modules directly
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    # Get state dictionaries for weight transfer
    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    # Split the original weights into two halves
    old_weight = state_dict["cv1.conv.weight"]
    half_channels = old_weight.shape[0] // 2
    state_dict_v2["cv0.conv.weight"] = old_weight[:half_channels]
    state_dict_v2["cv1.conv.weight"] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    # Split batch normalization parameters similarly
    for bn_key in ["weight", "bias", "running_mean", "running_var"]:
        old_bn = state_dict[f"cv1.bn.{bn_key}"]
        state_dict_v2[f"cv0.bn.{bn_key}"] = old_bn[:half_channels]
        state_dict_v2[f"cv1.bn.{bn_key}"] = old_bn[half_channels:]

    # Transfer remaining weights and buffers that don't need splitting
    for key in state_dict:
        if not key.startswith("cv1."):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes (like module parameters)
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and "_" not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    # Load the transferred state dictionary
    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    """
    Recursively replace all C2f modules with C2f_v2 modules in a neural network.
    
    This function traverses through all child modules of the given module and
    replaces any C2f instances with the improved C2f_v2 architecture while
    preserving all weights and parameters.
    
    Args:
        module (nn.Module): The root module to search for C2f modules
    """
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # Found a C2f module - replace it with C2f_v2
            # Infer the configuration from the first bottleneck module
            shortcut, is_CIB, lk = infer_shortcut(child_module.m[0])
            
            # Create new C2f_v2 module with same parameters
            c2f_v2 = C2f_v2(
                child_module.cv1.conv.in_channels,
                child_module.cv2.conv.out_channels,
                n=len(child_module.m),
                shortcut=shortcut,
                g=(
                    child_module.m[0].cv2.conv.groups
                    if not is_CIB
                    else child_module.cv2.conv.groups
                ),
                e=child_module.c / child_module.cv2.conv.out_channels,
                is_CIB=is_CIB,
                lk=lk,
            )
            
            # Transfer weights from old module to new module
            transfer_weights(child_module, c2f_v2)
            
            # Replace the module in the parent
            setattr(module, name, c2f_v2)
        else:
            # Recursively search in child modules
            replace_c2f_with_c2f_v2(child_module)
