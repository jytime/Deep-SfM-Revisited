import numpy as np
import os
import glob
import cv2
import pykitti

# Export Lidar Data to Depth Maps


# Change this to the directory where you store KITTI data
basedir = '/PATH/TO/YOUR/KITTIVO/DATA'


for sequence in ['00','01','02','03','04','05','06','07','08','09','10']:
    # Specify the dataset to load
    print('Sequence ' + sequence)

    # Load the data. Optionally, specify the frame range to load.
    # dataset = pykitti.odometry(basedir, sequence)
    dataset = pykitti.odometry(basedir, sequence)

    # for cam2_image in dataset.cam2:
    for i in range(len(dataset)):
        color = np.array(dataset.get_cam2(i))
        img_width = color.shape[1]
        img_height = color.shape[0]
        depth = np.zeros([img_height,img_width])

        velo = dataset.get_velo(i)
        velo[:,-1] = 1
        temp = dataset.calib.P_rect_20.dot(dataset.calib.T_cam0_velo)
        results = temp.dot(velo.T)

        uv = results[:2,:]/results[-1,:]
        z = results[-1,:]
        
        valid = (uv[0,:] > 0) & (np.round(uv[0,:]) < img_width) & (uv[1,:] > 0) &(np.round(uv[1,:]) < img_height) &(z>0)&(z<1000)
        valid_index = np.round(uv[:,valid]).astype('uint32')
        depth[valid_index[1],valid_index[0]] = z[valid]

        file_name = dataset.velo_files[i]
        out_name = file_name.replace('sequences','RealDepth').replace('bin','png')

        depth_to_write = depth.copy() * 256
        depth_to_write[depth_to_write<0] = 0
        depth_to_write[depth_to_write>65535] = 0
        depth_to_write = depth_to_write.astype('uint16')

        if not os.path.exists(os.path.dirname(out_name)):
            os.makedirs(os.path.dirname(out_name))

        cv2.imwrite(out_name,depth_to_write)

