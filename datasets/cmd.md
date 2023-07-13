<!--
 * @Description: 
 * @Author: Ming Liu (lauadam0730@gmail.com)
 * @Date: 2020-12-10 11:15:15
-->
cd /home/liuming/MIPAV/Method-SEG/codes

python train.py --gpu_rate=0.8 --epochs=150 --workers=2 --batch_size=4 --model=Unet --encoder=resnet18 --learning_rate=2e-4 --datatrain=train --savename=Unet;

python train.py --gpu_rate=0.8 --epochs=150 --workers=2 --batch_size=4 --model=AttUnet --encoder=resnet18 --learning_rate=2e-4 --datatrain=train --savename=AttUnet;

python train.py --gpu_rate=0.8 --epochs=150 --workers=2 --batch_size=4 --model=CENet --encoder=resnet18 --learning_rate=2e-4 --datatrain=train --savename=CENet-Res18;

python train.py --gpu_rate=0.8 --epochs=150 --workers=2 --batch_size=4 --model=CPFNet --encoder=resnet18 --learning_rate=2e-4 --datatrain=train --savename=CPFNet-Res18;

python train.py --gpu_rate=0.8 --epochs=150 --workers=2 --batch_size=4 --model=DeepLabV3 --encoder=resnet18 --learning_rate=2e-4 --datatrain=train --savename=DeepLabV3-Res18;

python train.py --gpu_rate=0.8 --epochs=150 --workers=2 --batch_size=4 --model=DeepLabV3Plus --encoder=resnet18 --learning_rate=2e-4 --datatrain=train --savename=DeepLabV3Plus-Res18;

python train.py --gpu_rate=0.8 --epochs=150 --workers=2 --batch_size=4 --model=DANet --encoder=resnet18 --learning_rate=2e-4 --datatrain=train --savename=DANet-Res18;

python train.py --gpu_rate=0.8 --epochs=150 --workers=2 --batch_size=4 --model=LinkNetSMP --encoder=resnet18 --learning_rate=2e-4 --datatrain=train --savename=LinkNetSMP-Res18;

python train.py --gpu_rate=0.8 --epochs=150 --workers=2 --batch_size=4 --model=UnetSMP --encoder=resnet18 --learning_rate=2e-4 --datatrain=train --savename=UnetSMP-Res18;

python train.py --gpu_rate=0.8 --epochs=150 --workers=2 --batch_size=4 --model=PANNetSMP --encoder=resnet18 --learning_rate=2e-4 --datatrain=train --savename=PANNetSMP-Res18;