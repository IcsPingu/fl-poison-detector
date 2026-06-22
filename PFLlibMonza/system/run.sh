@echo off
#conda activate pfllib
python main.py -nmc 30 -nc 100 -jr 1 -atk all -cc 6 -gr 300 -data Cifar10 -t 10 -ls 1 -did 1 -rfake 1 -m VGG
###python main.py -nm 0 -nc 100 -jr 1 -atk all -cc 5 -gr 300 -data Cifar10 -t 10 -ls 1 -did 1 -rfake 1
