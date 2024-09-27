import torch
from torch import linalg as LA
import copy
from transformers import PreTrainedModel

class ModelWithLPNorm(PreTrainedModel):
    def __init__(self, targetModule, baseModule=None, lambda_for_norm=0.05,  *args, **kwargs):
        super().__init__(config=targetModule.config, *args, **kwargs)
        self.targetModule=targetModule
        self.lambda_for_norm=lambda_for_norm
        self.baseModule=baseModule
        if self.baseModule is not None:
            for param in self.baseModule.parameters():
                param.requires_grad_(False)

    def forward(self, *args, **kwargs):

        result_from_target=self.targetModule.forward(*args, **kwargs)

        if self.baseModule is None:
            sum_norm=0.
            for param in self.targetModule.parameters():
                sum_norm += (LA.vector_norm(param)**2)

            act_loss=result_from_target.loss.clone()

            norm=sum_norm.clone()
            
            result_from_target.loss += (sum_norm * self.lambda_for_norm)
        else:
            sum_norm=0.
            for param_t, param_b in zip(self.targetModule.parameters(), self.baseModule.parameters()):
                sum_norm += (LA.vector_norm(param_t-param_b)**2)

            act_loss=result_from_target.loss.clone()

            norm=sum_norm.clone()

            result_from_target.loss += (sum_norm * self.lambda_for_norm)
        
        # act_loss is the language modeling loss, result_from_target.loss is the language modeling loss + damping term (backward on)
        return act_loss, norm, result_from_target
