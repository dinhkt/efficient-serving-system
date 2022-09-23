f=open("data.txt","r")
fw=open("profiler.txt","w")
lines=f.readlines()
for i,line in enumerate(lines):
    if i!=0:
        x=line.split(",")
        if x[0]=="1.8":
            x[0]="resnet18"
        if x[0]=="3.8":
            x[0]="resnet50"
        if x[0]=="15.3":
            x[0]="vgg16"
        if x[0]=="19.6":
            x[0]="vgg19"
        str=",".join(x)
        fw.write(str)
f.close()
fw.close()