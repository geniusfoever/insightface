import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re

file1 = open(r"G:\My Drive\insightface\model\r100-arcface-emore\log.txt", 'r')
Lines = file1.readlines()
if __name__=="__main__":
    epoch_list=[]
    train_acc_list=[]
    # Strips the newline character
    for line in Lines:

        if not re.search("Train-acc=",line):continue
        line=line.strip()
        # epoch=re.search("Bpoch.*]",line)
        # epoch=int(re.findall('\d+',epoch))
        # print(epoch)
        # epoch_list.append(epoch)
        acc=float(line.split("Train-acc=")[-1])
        train_acc_list.append(acc)

    plt.plot(train_acc_list)
    plt.show()


