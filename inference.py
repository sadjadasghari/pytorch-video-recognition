import torch
import numpy as np
from network import C3D_model
import cv2
torch.backends.cudnn.benchmark = True

def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/ucf_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    model = C3D_model.C3D(num_classes=101)
    checkpoint = torch.load('run/run_10/models/C3D-ucf101_epoch-99.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.train()

    # read video
    video = '/data/Sadjad/Datasets/ucf101/UCF-101/HandstandWalking/v_HandstandWalking_g22_c04.avi' # Haircut/v_Haircut_g03_c01.avi' # FrontCrawl/v_FrontCrawl_g07_c05.avi' # HulaHoop/v_HulaHoop_g02_c04.avi' # Biking/v_Biking_g06_c05.avi' # GolfSwing/v_GolfSwing_g11_c01.avi'
    # hmdb51/videos/dive/Extreme_Cliffdiving_dive_f_cm_np1_le_bad_2.avi'
    # ucf101/UCF-101/HighJump/v_HighJump_g07_c03.avi' # Skiing/v_Skiing_g09_c05.avi' # Archery/v_Archery_g01_c03.avi'
    #  '/data/Sadjad/Datasets/ucf101/UCF-101/Knitting/v_Knitting_g03_c05.avi'
    # '/data/Sadjad/Datasets/DALY/download_videos/videos/3\ WAYS\ OF\ APPLYING\ RED\ LIPSTICK\ l\ Pearltji-YCqSlzeFvn4.mp4'
    # '/data/Sadjad/Datasets/ucf101/UCF-101/StillRings/v_StillRings_g04_c02.avi'
    # capwrite = cv2.VideoCapture()
    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # MJPG')
    size = (
	int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
	int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	)
    out = cv2.VideoWriter('6.mp4',fourcc, 25, size)
    # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 1)
            cv2.putText(frame, "acc: %.4f" % probs[0][label], (10, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 1)
            clip.pop(0)

        out.write(frame)
        cv2.imshow('result', frame)
        #cv2.waitKey(30)

    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()









