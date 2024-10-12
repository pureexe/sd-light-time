def generate_tensorboard(self, batch, batch_idx, is_save_image=False, is_seperate_dir_with_epoch=False):
        self.pipe._callback_tensor_inputs = ["latents", "prompt_embeds", "latent_model_input", "timestep_cond", "added_cond_kwargs", "extra_step_kwargs"]
        USE_LIGHT_DIRECTION_CONDITION = True
        USE_NULL_TEXT = True

        is_apply_cfg = self.guidance_scale > 1
        
        # Apply the source light direction
        self.select_batch_keyword(batch, 'source')
        if USE_LIGHT_DIRECTION_CONDITION:
            set_light_direction(
                self.pipe.unet, 
                self.get_light_features(batch), 
                is_apply_cfg=False #during DDIM inversion, we don't want to apply the cfg
            )
        else:
            set_light_direction(
            self.pipe.unet, 
                None, 
                is_apply_cfg=False
            )

        # compute text embedding
        text_embbeding = get_text_embeddings(self.pipe, batch['text'])
        negative_embedding = get_text_embeddings(self.pipe, '')
        
        # get DDIM inversion
        ddim_latents = get_ddim_latents(
            self.pipe, batch['source_image'],
            text_embbeding,
            self.num_inversion_steps,
            torch.Generator().manual_seed(self.seed)
        )

        if USE_NULL_TEXT and self.guidance_scale > 1:
            null_embeddings, null_latents = get_null_embeddings(
                self.pipe,
                ddim_latents=ddim_latents,
                text_embbeding=text_embbeding,
                negative_embedding=negative_embedding,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inversion_steps,
                controlnet_image=self.get_control_image(batch) if hasattr(self.pipe, "controlnet") else None,
                num_null_optimization_steps=self.num_null_text_steps,
                generator=torch.Generator().manual_seed(self.seed)
            )            

        # if dataset is not list, convert to list
        for key in ['target_ldr_envmap', 'target_norm_envmap', 'target_image', 'target_sh_coeffs', 'word_name']:
            if key in batch and not isinstance(batch[key], list):
                batch[key] = [batch[key]]


        if USE_LIGHT_DIRECTION_CONDITION:
            #Apply the target light direction
            self.select_batch_keyword(batch, 'target')    
       

        # compute inpaint from nozie
        mse_output = []
        for target_idx in range(len(batch['target_image'])):
            set_light_direction(
                self.pipe.unet,
                self.get_light_features(batch, array_index=target_idx),
                is_apply_cfg=is_apply_cfg
            )
            
            if USE_NULL_TEXT and self.guidance_scale > 1:
                pt_image = apply_null_embedding(
                    self.pipe,
                    ddim_latents[-1],
                    null_embeddings,
                    text_embbeding,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inversion_steps,
                    generator=torch.Generator().manual_seed(self.seed),
                    controlnet_image=self.get_control_image(batch) if hasattr(self.pipe, "controlnet") else None,
                    null_latents=null_latents
                )
            else:
                zt_noise = ddim_latents[-1]
                pipe_args = {
                    "prompt_embeds": text_embbeding,
                    "negative_prompt_embeds": negative_embedding,
                    "latents": zt_noise.clone(),
                    "output_type": "pt",
                    "guidance_scale": self.guidance_scale,
                    "num_inference_steps": self.num_inference_steps,
                    "return_dict": False,
                    "generator": torch.Generator().manual_seed(self.seed)
                }
                if hasattr(self.pipe, "controlnet"):
                    pipe_args["image"] = self.get_control_image(batch)    
                pt_image, _ = self.pipe(**pipe_args)


            gt_on_batch = 'target_image' if "target_image" in batch else 'source_image'
            gt_image = (batch[gt_on_batch][target_idx] + 1.0) / 2.0
            tb_image = [gt_image, pt_image]

            if hasattr(self.pipe, "controlnet"):
                ctrl_image = self.get_control_image(batch)
                if isinstance(ctrl_image, list):
                    tb_image += ctrl_image
                else:
                    tb_image.append(ctrl_image)

            if hasattr(self, "pipe_chromeball"):
                with torch.inference_mode():
                    # convert pt_image to pil_image
                    to_inpaint_img = torchvision.transforms.functional.to_pil_image(pt_image[0].cpu())                
                    inpainted_image = inpaint_chromeball(to_inpaint_img,self.pipe_chromeball)
                    inpainted_image = torchvision.transforms.functional.to_tensor(inpainted_image).to(pt_image.device)
                    tb_image.append(inpainted_image[None])

            
            images = torch.cat(tb_image, dim=0)
            image = torchvision.utils.make_grid(images, nrow=int(np.ceil(np.sqrt(len(tb_image)))), normalize=True)
            
            
            tb_name = f"{batch['name'][0].replace('/','-')}/{batch['word_name'][target_idx][0].replace('/','-')}"
            
            self.logger.experiment.add_image(f'{tb_name}', image, self.global_step)

            # calcuarte psnr 
            mse = torch.nn.functional.mse_loss(gt_image, pt_image, reduction="none").mean()
            mse_output.append(mse[None])
            psnr = -10 * torch.log10(mse)
            self.log('psnr', psnr)
            if is_save_image:
                filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][target_idx][0].replace('/','-')}"
                epoch_text = f"epoch_{self.current_epoch:04d}/" if is_seperate_dir_with_epoch else ""

                # save with ground truth
                os.makedirs(f"{self.logger.log_dir}/{epoch_text}with_groudtruth", exist_ok=True)
                torchvision.utils.save_image(image, f"{self.logger.log_dir}/{epoch_text}with_groudtruth/{filename}.jpg")

                # save image file
                os.makedirs(f"{self.logger.log_dir}/{epoch_text}crop_image", exist_ok=True)
                torchvision.utils.save_image(pt_image, f"{self.logger.log_dir}/{epoch_text}crop_image/{filename}.jpg")
                # save psnr to file
                os.makedirs(f"{self.logger.log_dir}/{epoch_text}psnr", exist_ok=True)
                with open(f"{self.logger.log_dir}/{epoch_text}psnr/{filename}.txt", "w") as f:
                    f.write(f"{psnr.item()}\n")
                # save controlnet 
                if hasattr(self.pipe, "controlnet"):
                    os.makedirs(f"{self.logger.log_dir}/{epoch_text}control_image", exist_ok=True)
                    if isinstance(ctrl_image, list):
                        for i, c in enumerate(ctrl_image):
                            torchvision.utils.save_image(c, f"{self.logger.log_dir}/{epoch_text}control_image/{filename}_{i}.jpg")
                    else:
                        torchvision.utils.save_image(ctrl_image, f"{self.logger.log_dir}/{epoch_text}control_image/{filename}.jpg")
                # save the target_ldr_envmap
                if hasattr(batch, "target_ldr_envmap"):
                    os.makedirs(f"{self.logger.log_dir}/{epoch_text}target_ldr_envmap", exist_ok=True)
                    torchvision.utils.save_image(batch['target_ldr_envmap'][target_idx], f"{self.logger.log_dir}/{epoch_text}target_ldr_envmap/{filename}.jpg")
                if hasattr(batch, "target_norm_envmap"):
                    # save the target_norm_envmap
                    os.makedirs(f"{self.logger.log_dir}/{epoch_text}target_norm_envmap", exist_ok=True)
                    torchvision.utils.save_image(batch['target_norm_envmap'][target_idx], f"{self.logger.log_dir}/{epoch_text}target_norm_envmap/{filename}.jpg")
                if hasattr(batch, "source_ldr_envmap"):
                    # save the source_ldr_envmap
                    os.makedirs(f"{self.logger.log_dir}/{epoch_text}source_ldr_envmap", exist_ok=True)
                    torchvision.utils.save_image(batch['source_ldr_envmap'], f"{self.logger.log_dir}/{epoch_text}source_ldr_envmap/{filename}.jpg")
                if hasattr(batch, "source_norm_envmap"):
                    # save the source_norm_envmap
                    os.makedirs(f"{self.logger.log_dir}/{epoch_text}source_norm_envmap", exist_ok=True)
                    torchvision.utils.save_image(batch['source_norm_envmap'], f"{self.logger.log_dir}/{epoch_text}source_norm_envmap/{filename}.jpg")
                if hasattr(self, "pipe_chromeball"):
                    os.makedirs(f"{self.logger.log_dir}/{epoch_text}inpainted_image", exist_ok=True)
                    torchvision.utils.save_image(inpainted_image, f"{self.logger.log_dir}/{epoch_text}inpainted_image/{filename}.jpg")
                # save prompt
                os.makedirs(f"{self.logger.log_dir}/{epoch_text}prompt", exist_ok=True) 
                with open(f"{self.logger.log_dir}/{epoch_text}prompt/{filename}.txt", 'w') as f:
                    f.write(batch['text'][0])
                # save the source_image
                os.makedirs(f"{self.logger.log_dir}/{epoch_text}source_image", exist_ok=True)
                torchvision.utils.save_image(gt_image, f"{self.logger.log_dir}/{epoch_text}source_image/{filename}.jpg")
            if batch_idx == 0:
                if hasattr(self, "gate_trainable"):
                    # plot gate_trainable
                    for gate_id, gate in enumerate(self.gate_trainable):
                        self.logger.experiment.add_scalar(f'gate/{gate_id:02d}', gate, self.global_step)
                        self.logger.experiment.add_scalar(f'gate/average', gate, self.global_step)
            if self.global_step == 0:
                self.logger.experiment.add_text(f'text/{batch["word_name"][0]}', batch['text'][0], self.global_step)
            if self.global_step == 0 and batch_idx == 0:
                self.logger.experiment.add_text('learning_rate', str(self.learning_rate), self.global_step)
                if hasattr(self, "gate_multipiler"):
                    self.logger.experiment.add_text('gate_multipiler', str(self.gate_multipiler), self.global_step)
        mse_output = torch.cat(mse_output, dim=0)
        mse_output = mse_output.mean()
        return mse_output