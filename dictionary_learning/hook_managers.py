class HookManager:
    def __init__(self):
        """
        Hook manager for VLLM/HF models.
        
        This class manages hooks for extracting activations from VLLM/HF models.
        It provides methods to create, attach, and remove hooks, as well as
        to store and clear the extracted activations.
        
        Attributes:
            hooks_saved: List to store captured activations from hooks.
            input_ids: List to store input token IDs.
            handles: List of hook handles for later removal.
        """
        self.hooks_saved = []  # Stores captured activations
        self.input_ids = []    # Stores input token IDs
        #self.attn_masks = []   # Stores attention masks
        self.handles = []      # Stores hook handles for later removal

    def create_activation_hook(self, io='out'):
        def activation_hook(module, input, output):
            if io == 'out':
                if isinstance(output, tuple):
                    output = output[0]
                else:
                    output = output
                if len(output.shape) == 2:
                    # Add position dimension for classifier
                    output = output.unsqueeze(1)
            elif io == 'in':
                if isinstance(input, tuple):
                    output = input[0]
                else:
                    output = input
            self.hooks_saved.append(output.detach())
        return activation_hook

    def attach_and_verify_hook(self, submodule, io='out'):
        hook = self.create_activation_hook(io)
        handle = submodule.register_forward_hook(hook)
        self.handles.append(handle)
        return handle
    
    # def create_input_activation_hook(self):
    #     def activation_hook(module, input, output):
    #         self.input_ids.append(input[0].detach())
    #     return activation_hook

    
    # def attach_input_hook(self, model):
    #     if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
    #         layer_to_hook = model.transformer.embed_tokens
    #     elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
    #         layer_to_hook = model.model.embed_tokens
    #     elif hasattr(model, 'language_model'):
    #         layer_to_hook = model.language_model.model.embed_tokens
    #     else:
    #         raise AttributeError(
    #             f"Unsupported model architecture. Model type: {type(model)}. "
    #             f"Available attributes: {dir(model)}"
    #         )
    #     hook_input = self.create_input_activation_hook()
    #     handle = layer_to_hook.register_forward_hook(hook_input)
    #     self.handles.append(handle)
    #     return handle


    def clear_saved_data(self):
        """Clear saved activations and input IDs without removing hooks"""
        self.hooks_saved.clear()
        self.input_ids.clear()

    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
