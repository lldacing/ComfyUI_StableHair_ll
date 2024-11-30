[中文文档](README.md)

Stable-Hair: Real-World Hair Transfer via Diffusion Model

## Preview
![save api extended](doc/image.png)

## [Workflow example](example/workflow_base.png)
In the workflow, it is the simplest replacement without pasting back the original image. The complete process is: crop the avatar area containing hair ->generate a bald head ->migrate the reference image hair ->paste the result image back to the original image

### tips：
- The width and height of the images for generating bald heads and transferring hair need to be multiples of 8, the two cropped image should be same size, and they should be front facing photos
- Choose the SD1.5 model

## Install

- Manual
```shell
    cd custom_nodes
    git clone https://github.com/lldacing/ComfyUI_StableHair_ll.git
    cd ComfyUI_StableHair_ll
    # restart ComfyUI
```
    

## Model
From [HuggingFace](https://huggingface.co/lldacing/StableHair/tree/main) download all files to `ComfyUI/models/diffusers/StableHair` directory.

Suggest using huggingface-cli to download
```
# Start the command line in the ComfyUI/models directory and execute the following command
huggingface-cli download lldacing/StableHair --local-dir StableHair
```
The directory structure is as follows:
```
ComfyUI
  └─models
      └─diffusers
          └─StableHair
              └─hair_encoder_model.bin
              └─hair_adapter_model.bin
              └─hair_controlnet_model.bin
              └─hair_bald_model.bin
```


## Thanks

Original Project [Xiaojiu-z/Stable-Hair](https://github.com/Xiaojiu-z/Stable-Hair)


