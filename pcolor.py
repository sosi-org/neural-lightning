"""
pcolor: for plotting pcolor using matplotlib
"""

from utils import linux_plot_issue, is_linux, is_mac

import matplotlib.pyplot as plt
import numpy as np
import os
import time

output_directory = './generated'
os.makedirs(output_directory, exist_ok=True)

class PColor:
    """ Show and save pcolor (w,h,3) in range float [0,1] """
    @staticmethod
    def plot_show_image(G_paintings2d, file_id, sleep_sec, more_info):
        """
        more_info = (accuracy, score)
        """
        plt.clf()

        import matplotlib
        matplotlib.rc('axes', edgecolor='white')
        matplotlib.rc('axes', facecolor='black')

        ax = plt.gca()
        ax.set_facecolor((0.0, 0.0, 0.0))
        #print(dir(ax))
        #exit()
        #ax.set_edgecolor((1.0, 1.0, 1.0))

        #print(np.max(np.max(G_paintings2d,axis=2), axis=0))
        #print(np.min(np.min(G_paintings2d,axis=2), axis=1))
        #print(G_paintings2d.shape)
        #plt.imshow(G_paintings2d)
        #plt.imshow((G_paintings2d * 0.2 + 0.5)*0.2)
        #img_pix_rescale = (G_paintings2d * 0.05 + 0.5)
        #img_pix_rescale = (G_paintings2d)
        #plt.imshow(img_pix_rescale, vmin=-100, vmax=100)
        #img_pix_rescale = ((G_paintings2d) / 80.0 *40  ) +0.5
        #img_pix_rescale = ((G_paintings2d) / 2.0  ) +0.5
        img_pix_rescale = G_paintings2d
        # print('img_pix_rescale.shape', img_pix_rescale.shape)
        RGB3D = 3
        assert len(img_pix_rescale.shape) == RGB3D
        if img_pix_rescale.shape[2] < RGB3D:
            img_pix_rescale = np.max(img_pix_rescale, axis=2)
            img_pix_rescale = img_pix_rescale[:,:,None]
            img_pix_rescale = np.repeat(img_pix_rescale, RGB3D, axis=2)
        if img_pix_rescale.shape[2] > RGB3D:
            img_pix_rescale = img_pix_rescale[:,:,:RGB3D]

        #scaled_back_to_255 = img_pix_rescale * 128
        #scaled_back_to_255 = ((img_pix_rescale / 2.0)+0.5) * 128
        scaled_back_to_255 = img_pix_rescale * 127.0 + 128
        scaled_back_to_255[scaled_back_to_255 > 255] = 255
        plt.imshow(scaled_back_to_255.astype(np.uint8))
        print('min max:', np.min(img_pix_rescale.ravel()), np.max(img_pix_rescale.ravel()))
        #plt.pcolor(np.mean(G_paintings2d, axis=2))
        # acc, score = more_info
        # text1 = 'D accuracy=%.2f (0.5 for D to converge)' % acc
        # text2 = 'D score= %.2f (-1.38 for G to converge)' % score
        text1, text2 = more_info
        plt.text(-.5, 0, text1, fontdict={'size': 15})
        plt.text(-.5, G_paintings2d.shape[1]*0.5, text2, fontdict={'size': 15})
        # plt.colorbar()


        PColor.next_plot(sleep_sec)

        if(file_id is not None):
            PColor.save( os.path.join(output_directory, file_id + '.png') )

    @staticmethod
    def save(filename):
        plt.draw()
        plt.savefig( filename )
        print("saved")
        if is_mac():
            wait_time_sec = 0.1
            time.sleep(wait_time_sec)

    """ Next plot. Platform-independent """
    @staticmethod
    def next_plot(sleep_sec):
        if is_mac():
            print('draw')
            import sys
            sys.stdout.flush()

            plt.draw()
            time.sleep(sleep_sec)
        elif is_linux():
            # """ "Modal" """
            # plt.show()


            #plt.draw()
            #plt.show(block=False)
            #time.sleep(0.5)
            #plt.draw()
            """
            # futile:
            plt.ion()
            plt.draw()
            plt.show()
            plt.ioff()

            time.sleep(sleep_sec)
            time.sleep(2.0)
            plt.close()
            plt.ioff()
            """
        else:
            raise

    @staticmethod
    def init():
        linux_plot_issue()

        plt.cla()
        #plt.imshow(main_artworks[0])

        if is_linux():
            # plt.ioff()  # not necessary
            # plt.show()

            #plt.ion()
            plt.draw()
            plt.show(block=False)
            plt.draw()
            time.sleep(0.5)
            return
        elif is_mac():
            plt.draw()
            plt.ion()
            plt.show()
            time.sleep(0.1)
            plt.ion()   # something about continuous plotting
            return
        else:
            raise
        raise

    @staticmethod
    def last(self):
        if is_mac():
            plt.ioff()
            plt.show()
        elif is_linux():
            pass
        else:
            raise

# [1] This class is from https://github.com/sosi-org/neural-networks-sandbox/blob/4cba7254b52551c9bd4235e2f6d41feb3e1c8447/glyphnet/utils/pcolor.py
