from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline, AutoencoderKL, IFImg2ImgPipeline

class PureIFPipeline(IFPipeline):
    def set_initial_image(self, image):
        self.initial_image = image

    def prepare_intermediate_images(self, *args, **kwargs):
        if hasattr(self, 'initial_image') and self.initial_image is not None:
            return self.initial_image
        intermediate_images = super().prepare_intermediate_images(*args, **kwargs)
        return intermediate_images

