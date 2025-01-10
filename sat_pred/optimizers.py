import torch
from sat_pred.loss import LossFunction

class AdamW:
    """AdamW optimizer"""

    def __init__(self, lr=0.0005, **kwargs):
        """AdamW optimizer"""
        self.lr = lr
        self.kwargs = kwargs

    def __call__(self, model):
        """Return optimizer"""
        return torch.optim.AdamW(model.parameters(), lr=self.lr, **self.kwargs)

    
class AdamWReduceLROnPlateau:
    """AdamW optimizer and reduce on plateau scheduler"""

    def __init__(
        self, lr=0.0005, patience=10, factor=0.2, threshold=2e-4, step_freq=None, **opt_kwargs
    ):
        """AdamW optimizer and reduce on plateau scheduler"""
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.step_freq = step_freq
        self.opt_kwargs = opt_kwargs

    def __call__(self, model):

        opt = torch.optim.AdamW(
            model.parameters(), lr=self.lr, **self.opt_kwargs
        )

        if isinstance(model.target_loss, str):
            monitor = f"{model.target_loss}/val"
        elif isinstance(model.target_loss, LossFunction):
            monitor = f"{model.target_loss.name}/val"
        else:
            raise ValueError(f"Unknown loss type: {type(model)}")

        sch = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
            ),
            "monitor": monitor,
        }

        return [opt], [sch]
    

class AdamWReduceLROnPlateauGroups:
    """AdamW optimizer and reduce on plateau scheduler with support for parameter groups"""

    def __init__(
        self,
        lr=0.0005,
        patience=10,
        factor=0.2,
        threshold=2e-4,
        step_freq=None,
        param_groups=None,
        **opt_kwargs
    ):
        """
        Args:
            lr (float): Base learning rate
            patience (int): Number of epochs with no improvement after which learning rate will be reduced
            factor (float): Factor by which the learning rate will be reduced
            threshold (float): Threshold for measuring the new optimum
            step_freq (Optional[int]): Step frequency
            param_groups (Optional[List[dict]]): List of parameter group configurations.
                Each dict should contain:
                - 'params_pattern': str or list of str, patterns to match parameter names
                - 'lr_multiplier': float, multiplier for the base learning rate
            **opt_kwargs: Additional optimizer kwargs
        """
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.step_freq = step_freq
        self.param_groups = param_groups or []
        self.opt_kwargs = opt_kwargs

    def __call__(self, model):
        # Create parameter groups
        if not self.param_groups:
            # If no parameter groups specified, use single group with base learning rate
            parameters = model.parameters()
        else:
            parameters = self._create_param_groups(model)

        opt = torch.optim.AdamW(
            parameters,
            lr=self.lr,
            **self.opt_kwargs
        )

        if isinstance(model.target_loss, str):
            monitor = f"{model.target_loss}/val"
        elif isinstance(model.target_loss, LossFunction):
            monitor = f"{model.target_loss.name}/val"
        else:
            raise ValueError(f"Unknown loss type: {type(model)}")

        sch = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
            ),
            "monitor": monitor,
        }

        return [opt], [sch]

    def _create_param_groups(self, model):
        """Create parameter groups based on name patterns"""
        # Keep track of parameters that have been assigned to groups
        assigned_params = set()
        param_groups = []

        # Create groups based on patterns
        for group_config in self.param_groups:
            patterns = group_config['params_pattern']
            if isinstance(patterns, str):
                patterns = [patterns]

            group_params = []
            for name, param in model.named_parameters():
                if any(pattern in name for pattern in patterns):
                    group_params.append(param)
                    assigned_params.add(name)

            if group_params:
                param_groups.append({
                    'params': group_params,
                    'lr': self.lr * group_config['lr_multiplier']
                })

        # Create default group for remaining parameters
        default_params = [
            param for name, param in model.named_parameters()
            if name not in assigned_params
        ]
        if default_params:
            param_groups.append({
                'params': default_params,
                'lr': self.lr
            })

        return param_groups