export CUDA_VISION_DEVICE=0,1,2,3,4,5,6,7
#export CUDA_VISION_DEVICE=0,1
#echo ${CUDA_VISION_DEVICE}
cd /data1/dsq/
#python train.py  --multiprocessing-distributed  --gpu=0 --rank=0  --world-size=1 -a resnet18  -q DSQ --quan_bit 8 --dist-url=tcp://172.30.37.1:59202 /dataset/public/ImageNetOrigin/
#python train.py  --multiprocessing-distributed  --gpu=0 --rank=0  --world-size=1 -a resnet18  -q DSQ --quan_bit 8 --dist-url="file:///data1/dsq/1.txt" /dataset/public/ImageNetOrigin/
python train.py --multiprocessing-distributed=True  --gpu=0 --rank=0  --world-size=1 --dist-url="tcp://localhost:54078" --data="/dataset/public/ImageNetOrigin/" -a resnet18  -q DSQ --quan_bit 8  --log_path=./log_withnot_quant_input
python train.py --multiprocessing-distributed=True  --gpu=0 --rank=0  --world-size=1 --dist-url="tcp://localhost:54078" --data="/dataset/public/ImageNetOrigin/" -a resnet18  -q DSQ --quan_bit 8 --quantize_input --log_path=./log_with_quant_input
python train.py --multiprocessing-distributed=True  --gpu=0 --rank=0  --world-size=1 --dist-url="tcp://localhost:54078" --data="/dataset/public/ImageNetOrigin/" -a resnet34  -q DSQ --quan_bit 4  --log_path=./log_withnot_quant_input_int4_res34
python train.py --multiprocessing-distributed=True  --gpu=0 --rank=0  --world-size=1 --dist-url="tcp://localhost:54078" --data="/dataset/public/ImageNetOrigin/" -a resnet34  -q DSQ --quan_bit 4 --quantize_input --log_path=./log_with_quant_input_int4_res34
python train.py --multiprocessing-distributed=True  --gpu=0 --rank=0  --world-size=1 --dist-url="tcp://localhost:54078" --data="/dataset/public/ImageNetOrigin/" -a resnet18  -q DSQ --quan_bit 4  --log_path=./log_withnot_quant_input_int4
python train.py --multiprocessing-distributed=True  --gpu=0 --rank=0  --world-size=1 --dist-url="tcp://localhost:54078" --data="/dataset/public/ImageNetOrigin/" -a resnet18  -q DSQ --quan_bit 4 --quantize_input --log_path=./log_with_quant_input_int4
python train.py --multiprocessing-distributed=True  --gpu=0 --rank=0  --world-size=1 --dist-url="tcp://localhost:54078" --data="/dataset/public/ImageNetOrigin/" -a resnet18  -q DSQ --quan_bit 2  --log_path=./log_without_quant_input_int2
python train.py --multiprocessing-distributed=True  --gpu=0 --rank=0  --world-size=1 --dist-url="tcp://localhost:54078" --data="/dataset/public/ImageNetOrigin/" -a resnet18  -q DSQ --quan_bit 2 --quantize_input --log_path=./log_with_quant_input_int2
#python train.py  --multiprocessing-distributed  --gpu=0 --rank=0  --world-size=1 -a resnet18  -q DSQ --quan_bit 8 --dist-url="tcp://localhost:54078" --data="/dataset/public/ImageNetOrigin/" --log_path=./log_resnet18_ImageNet_int8_DSQ/log_withnot_quant_input
#python train.py --multiprocessing-distributed  --gpu=0 --rank=0  --world-size=1 --dist-url="tcp://localhost:54078" --data="/dataset/public/ImageNetOrigin/" -a resnet18  -q DSQ --quan_bit 8 --quantize_input --log_path=./log_resnet18_ImageNet_int8_DSQ/log_with_quant_input
#python train.py   -a resnet18  -q DSQ --quan_bit 8  --data="/dataset/public/ImageNetOrigin/" --log_path=./log_resnet18_ImageNet_int8_DSQ/log_withnot_quant_input