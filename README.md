# ViT_Inversion

## Settings
```shell
$ pip install -r requirements.txt
```
### Load the pretrained moco models for GradVit (gv)
* download .tar files on link https://huggingface.co/sotaBrewer824/x4ssl/commit/dd22d2b3d14aaa97a5703bc54db5f34c1044e2af
and move it on the demodata folder.
## Run Inversion Attacks on the ViT models.
```shell
$ python run_attack_imagenet.py --arch vit -a gv --lr 0.1 --max-iters 20000 -b 8 -g 1 
$ python run_attack_imagenet.py --arch vit -a gi --lr 0.1 --max-iters 20000 -b 8 -g 1 
$ python run_attack_imagenet.py --arch vit -a idlg --lr 0.1 --max-iters 20000 -b 8 -g 1 
$ python run_attack_imagenet.py --arch vit -a dlg --lr 1.0 --max-iters 20000 -b 8 -g 1
$ python run_attack_imagenet.py --arch vit -a gs --lr 1.0 --max-iters 20000 -b 8 -g 1
$ python run_attack_imagenet.py --arch vit -a cpl --lr 1.0 --max-iters 20000 -b 8 -g 1
```

## A list of attacks we can do
`attack mode 'dlg', 'idlg', 'gs', 'cpl', 'gi', 'gv'`
- **'Deep Leakage from Gradients'** [[DLG](https://arxiv.org/abs/1906.08935)]
- **iDLG: Improved Deep Leakage from Gradients** [[iDLG](https://arxiv.org/pdf/2001.02610)]
- **See through Gradients: Image Batch Recovery via GradInversion** [[gi](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.pdf)]
- **Inverting Gradients - How easy is it to break privacy in federated learning?** [[gs](https://arxiv.org/abs/2003.14053)]
- **GradViT: Gradient Inversion of Vision Transformers** [[gv](https://arxiv.org/abs/2203.11894)]

## A list of model architecture we can do
- ResNet18
- ResNet50
- LeNet
- ConvNet
- Vision Transformer