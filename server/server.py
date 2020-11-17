import socket
import torchvision
import torch
import numpy as np


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.cuda()
model.eval()

def forward_image_list(input_tensor_list):
    pred_list = model(input_tensor_list)
    mask_list = []
    for pred in pred_list:
        pred_score = list(pred['scores'].detach().cpu().numpy())
        pred_class = list(pred['labels'].detach().cpu().numpy())
        select_ind = [pred_score.index(x) for x, label in zip(pred_score, pred_class) if x > 0.9 and label == 1]
        masks = pred['masks']
        select_mask = masks[select_ind, :, :, :] > 0.3
        total_mask = torch.sum(select_mask, dim=0).float()
        total_mask = (total_mask>=1).int()*255
        mask_list.append(total_mask)
    return mask_list


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.bind(('127.0.0.1', 2222))
s.listen(5)
print("waiting...")

rescale = 1
height = int(500 * rescale)
width = int(1200 * rescale)
lenth = width*height*3

while True:
    sock, addr = s.accept()
    while True:
        data = sock.recv(1024)
        if len(data)>0:
            total_data = data
            while len(total_data)<lenth and len(data)>0:
                data = sock.recv(1024)
                total_data += data
                # print(len(total_data))
            # total_data recv finished
            np_array = np.frombuffer(total_data, dtype=np.uint8)
            print(np_array.shape)
            input_tensor = torch.from_numpy(np_array).float().view((height, width, 3))
            input_tensor = input_tensor.permute((2, 0, 1))
            input_tensor = input_tensor/255
            mask_list = forward_image_list([input_tensor.cuda()])
            mask0_numpy = mask_list[0].detach().cpu().numpy().astype(np.uint8)
            mask0_numpy_bytes = mask0_numpy.tobytes()
            sock.sendall(mask0_numpy_bytes)
            print("send mask bytes!" + str(len(mask0_numpy_bytes)))
        else:
            break