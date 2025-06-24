from vll_datasets.DiffusionRendererEnvmapDataset import DiffusionRendererEnvmapDataset

LIGHT_KEY = ['light_ldr', 'light_log_hdr', 'light_dir', "irradiant_ldr", 'irradiant_log_hdr', 'irradiant_dir', 'envmap_feature_layer0', 'envmap_feature_layer1', 'envmap_feature_layer2', 'envmap_feature_layer3']

class DiffusionRendererEnvmapDDIMDataset(DiffusionRendererEnvmapDataset):
    def __init__(self, root_dir="/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/val", index_file=None,  *args, **kwargs):
        super().__init__(root_dir, index_file,  *args, **kwargs)

    def get_item(self, idx):
        output = super().get_item(idx)

        # change the light source to add "source_" prefix

        for key in LIGHT_KEY:
            if key in output:
                if key in output:
                    output[f'source_{key}'] = output.pop(key)

        # add source and target image if image is in the output 
        if 'image' in output:
            output['source_image'] = output.pop('image')            
            output['target_image'] = []

        # for target lighting, we will use target_ prefix
        # here is the thing that will output in format of array 
        for key in LIGHT_KEY:
            if 'source_' + key in output:
                output[f'target_{key}'] = []
        output['envmap_name'] = []

        for envmap_name in self.envmap_index[idx]:
            if 'target_light_ldr' in output or 'target_light_log_hdr' in output or 'target_light_dir' in output:
                light = self.get_light(envmap_name)
                for key in LIGHT_KEY:
                    if 'target_' + key in output:
                        output[f'target_{key}'].append(light[key])
            
            if 'target_envmap_feature_layer0' in output or 'target_envmap_feature_layer1' in output or 'target_envmap_feature_layer2' in output or 'target_envmap_feature_layer3' in output:
                envmap_feature = self.get_envmap_feature(envmap_name)
                for i in range(4):
                    if f'target_envmap_feature_layer{i}' in output:
                        output[f'target_envmap_feature_layer{i}'].append(envmap_feature[i])
            
            if 'target_image' in output:
                output['target_image'].append(self.get_image(envmap_name))

            output['envmap_name'].append(envmap_name)

        return output
    
