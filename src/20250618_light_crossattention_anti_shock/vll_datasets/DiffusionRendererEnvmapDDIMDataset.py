from vll_datasets.DiffusionRendererEnvmapDataset import DiffusionRendererEnvmapDataset

LIGHT_KEY = ['light_ldr', 'light_log_hdr', 'light_dir', "irradiant_ldr", 'irradiant_log_hdr', 'irradiant_dir']

class DiffusionRendererEnvmapDDIMDataset(DiffusionRendererEnvmapDataset):
    def __init__(self, root_dir="/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/val", index_file=None,  *args, **kwargs):
        super().__init__(root_dir, index_file,  *args, **kwargs)

    def get_item(self, idx):
        output = super().get_item(idx)

        # if light is not in the components, we will return the output as it is
        if not 'light' in self.components:        
            return output

        # change the light source to add "source_" prefix

        for key in LIGHT_KEY:
            if key in output:
                if key in output:
                    output[f'source_{key}'] = output.pop(key)

        # for target lighting, we will use target_ prefix
        # here is the thing that will output in format of array 
        for key in LIGHT_KEY:
            if key in output:
                output[f'target_{key}'] = []
        output['envmap_name'] = []

        for envmap_name in self.envmap_index[idx]:
            light = self.get_light(envmap_name)
            for key in LIGHT_KEY:
                if 'target_' + key in output:
                    output[f'target_{key}'].append(light[key])
            output['envmap_name'].append(envmap_name)

        return output
    
