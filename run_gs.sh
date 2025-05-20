#!/bin/bash
python run_attack_imagenet.py --arch vit -a gv --lr 0.1 --max-iters 20000 -b 8 -g 1 ;
python run_attack_imagenet.py --arch lenet -a gv --lr 0.1 --max-iters 20000 -b 8 -g 1 ;
python run_attack_imagenet.py --arch convnet -a gv --lr 0.1 --max-iters 20000 -b 8 -g 1 ;
python run_attack_imagenet.py --arch resnet18 -a gv --lr 0.1 --max-iters 20000 -b 8 -g 1 ;
python run_attack_imagenet.py --arch resnet50 -a gv --lr 0.1 --max-iters 20000 -b 8 -g 1 ;

python run_attack_imagenet.py --arch lenet -a gi --lr 0.1 --max-iters 20000 -b 8 -g 1 ;
python run_attack_imagenet.py --arch convnet -a gi --lr 0.1 --max-iters 20000 -b 8 -g 1 ;
python run_attack_imagenet.py --arch resnet18 -a gi --lr 0.1 --max-iters 20000 -b 8 -g 1 ;
python run_attack_imagenet.py --arch resnet50 -a gi --lr 0.1 --max-iters 20000 -b 8 -g 1 ;
python run_attack_imagenet.py --arch vit -a gi --lr 0.1 --max-iters 20000 -b 8 -g 1 ;

python run_attack_imagenet.py --arch resnet18 -a idlg --lr 0.1 --max-iters 20000 -b 8 -g 1  ;
python run_attack_imagenet.py --arch resnet50 -a idlg --lr 0.1 --max-iters 20000 -b 8 -g 1  ;
python run_attack_imagenet.py --arch vit -a idlg --lr 0.1 --max-iters 20000 -b 8 -g 1  ;
python run_attack_imagenet.py --arch lenet -a idlg --lr 0.1 --max-iters 20000 -b 8 -g 1 ;
python run_attack_imagenet.py --arch convnet -a idlg --lr 0.1 --max-iters 20000 -b 8 -g 1 ;

python run_attack_imagenet.py --arch resnet18 -a dlg --lr 1.0 --max-iters 20000 -b 8 -g 1;
python run_attack_imagenet.py --arch resnet50 -a dlg --lr 1.0 --max-iters 20000 -b 8 -g 1;
python run_attack_imagenet.py --arch vit -a dlg --lr 1.0 --max-iters 20000 -b 8 -g 1;
python run_attack_imagenet.py --arch lenet -a dlg --lr 1.0 --max-iters 20000 -b 8 -g 1;
python run_attack_imagenet.py --arch convnet -a dlg --lr 1.0 --max-iters 20000 -b 8 -g 1;

python run_attack_imagenet.py --arch resnet18 -a gs --lr 0.1 --max-iters 20000 -b 1 -g 1; #--with-labels
python run_attack_imagenet.py --arch resnet50 -a gs --lr 1.0 --max-iters 20000 -b 8 -g 1;
python run_attack_imagenet.py --arch vit -a gs --lr 1.0 --max-iters 20000 -b 8 -g 1;
python run_attack_imagenet.py --arch lenet -a gs --lr 1.0 --max-iters 20000 -b 8 -g 1;
python run_attack_imagenet.py --arch convnet -a gs --lr 1.0 --max-iters 20000 -b 8 -g 1;

python run_attack_imagenet.py --arch resnet18 -a cpl --lr 1.0 --max-iters 20000 -b 8 -g 1;
python run_attack_imagenet.py --arch resnet50 -a cpl --lr 1.0 --max-iters 20000 -b 8 -g 1;
python run_attack_imagenet.py --arch vit -a cpl --lr 1.0 --max-iters 20000 -b 8 -g 1;
python run_attack_imagenet.py --arch lenet -a cpl --lr 1.0 --max-iters 20000 -b 8 -g 1;

## test 8
python run_attack_imagenet.py --arch convnet -a cpl --lr 1.0 --max-iters 20000 -b 8 -g 1;

# python run_attack_imagenet.py --arch resnet18 -a dlg --lr 1.0 --max-iters 20000 -b 8 
# python run_attack_imagenet.py --arch resnet50 -a dlg --lr 1.0 --max-iters 20000 -b 8 
# python run_attack_imagenet.py --arch vit -a dlg --lr 1.0 --max-iters 20000 -b 8 
# python run_attack_imagenet.py --arch lenet -a dlg --lr 1.0 --max-iters 20000 -b 8 
# python run_attack_imagenet.py --arch convnet -a dlg --lr 1.0 --max-iters 20000 -b 8 | tee ./log/logfile_$(date '+%Y-%m-%d-%H:%M').log 



#python run_attack_imagenet.py --arch resnet18 -a gs --lr 0.001 --max-iters 20000 -b 8 -g 1;
# python run_attack_imagenet.py --arch resnet50 -a gs --lr 0.001 --max-iters 20000 -b 8 -g 1;
# python run_attack_imagenet.py --arch vit -a gs --lr 0.001 --max-iters 20000 -b 8 -g 1;
# python run_attack_imagenet.py --arch lenet -a gs --lr 0.001 --max-iters 20000 -b 8 -g 1;
# python run_attack_imagenet.py --arch convnet -a gs --lr 0.001 --max-iters 20000 -b 8 -g 1;




# python run_attack_imagenet.py --arch resnet18 -a idlg --lr 0.001 --max-iters 20000 -b 8 -g 1 ;
# python run_attack_imagenet.py --arch resnet50 -a idlg --lr 0.001 --max-iters 20000 -b 8 -g 1 ;
# python run_attack_imagenet.py --arch vit -a idlg --lr 0.001 --max-iters 20000 -b 8 -g 1 ;
# python run_attack_imagenet.py --arch lenet -a idlg --lr 0.001 --max-iters 20000 -b 8 -g 1 ;
# python run_attack_imagenet.py --arch convnet -a idlg --lr 0.001 --max-iters 20000 -b 8 -g 1  ;
