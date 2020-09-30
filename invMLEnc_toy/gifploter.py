import matplotlib.pyplot as plt
import imageio
import os
import numpy as np


class GIFPloter():
    def __init__(self, args, model):

        self.plot_method = 'Li'

        self.gif_axlist = []
        self.clist = ['r', 'g', 'b', 'y', 'm', 'c', 'k',
                      'pink', 'lightblue', 'lightgreen', 'grey']
        self.fig, self.ax = plt.subplots()
        self.his_loss = None
        self.NetworkStructure = args['NetworkStructure']
        self.current_subfig_index = 2
        self.plot_every_epoch = args['PlotForloop']

        self.infor_index_list = model.plot_index_list
        self.name_list = model.name_list
        self.num_subfig = len(model.plot_index_list)
        self.layer_num = len(args['NetworkStructure']) - 1

        if self.plot_method == 'Zang':
            self.num_fig_every_row = int(np.sqrt(self.num_subfig))+1
            self.num_row = int(1+(self.num_subfig - 0.5) //
                               self.num_fig_every_row)
            self.sub_position_list = [i+1 for i in range(self.num_subfig)]
        if self.plot_method == 'Li':
            self.num_fig_every_row = 2
            self.num_row = int(1+(self.num_subfig - 0.5) //
                               self.num_fig_every_row)
            self.sub_position_list = [i*2 + 1 for i in range(self.num_subfig//2)] +\
                                     [self.num_subfig] + \
                list(reversed([i*2 + 2 for i in range(self.num_subfig//2)]))


    def PlotOtherLayer(self, fig,
                       data, label,
                       title='',
                       fig_position0=1,
                       fig_position1=1,
                       fig_position2=1,
                       s=8):
        from sklearn.decomposition import PCA
        # input(fig_position)

        color_list = []
        for i in range(label.shape[0]):
            color_list.append(int(label[i]))

        if data.shape[1] > 3:
            pca = PCA(n_components=2)
            data_em = pca.fit_transform(data)
        else:
            data_em = data

        data_em = data_em-data_em.mean(axis=0)

        if data_em.shape[1] == 3:
            ax = fig.add_subplot(fig_position0, fig_position1,
                                 fig_position2, projection='3d')

            ax.scatter(
                data_em[:, 0], data_em[:, 1], data_em[:, 2],
                c=color_list, s=s, cmap='rainbow')

        if data_em.shape[1] == 2:
            ax = fig.add_subplot(fig_position0, fig_position1, fig_position2)
            ax.scatter(
                data_em[:, 0], data_em[:, 1], c=label, s=s, cmap='rainbow')
            plt.axis('equal')

        plt.title(title)
        self.current_subfig_index = self.current_subfig_index+1
    
    def update_loss(self, loss=None):
        """ 0721, append loss list """
        if self.his_loss is None and loss is not None:
            self.his_loss = [[] for i in range(len(loss))]
        elif loss is not None:
            for i, loss_item in enumerate(loss):
                self.his_loss[i].append(loss_item)

    def AddNewFig(self, output_info, label_point, loss=None, title_='', save=True):

        self.update_loss(loss)

        self.current_subfig_index = 1
        fig = plt.figure(figsize=(5*self.num_fig_every_row, 5*self.num_row))

        for i, index in enumerate(self.infor_index_list):
            self.PlotOtherLayer(
                fig, output_info[index],
                label_point, title=self.name_list[index],
                fig_position0=self.num_row,
                fig_position1=self.num_fig_every_row,
                fig_position2=int(self.sub_position_list[i]))

        if loss is not None:
            loss_interval = 200
            loss_sum = []
            for i in range(len(self.his_loss[1])):
                tmp = 0
                for j in range(len(self.his_loss)):
                    try:
                        tmp += self.his_loss[j][i]
                    except:
                        pass
                loss_sum.append(tmp)
                
            ax = fig.add_subplot(self.num_row, self.num_fig_every_row,
                                int(max(self.sub_position_list))+1)
            l1, = ax.plot(
                [i*loss_interval for i in range(len(self.his_loss[0]))],
                self.his_loss[0], 'bo-')
            l2, = ax.plot(
                [i*loss_interval for i in range(len(self.his_loss[0]))],
                self.his_loss[1], 'ko-')
            l3, = ax.plot(
                [i*loss_interval for i in range(len(self.his_loss[0]))],
                self.his_loss[2], 'yo-')
            l4, = ax.plot(
                [i*loss_interval for i in range(len(self.his_loss[0]))],
                self.his_loss[3], 'ro-')
            l5, = ax.plot(
                [i*loss_interval for i in range(len(self.his_loss[0]))],
                self.his_loss[4], 'mo-')
            l6, = ax.plot(
                [i*loss_interval for i in range(len(self.his_loss[0]))],
                self.his_loss[5], 'go-')
            l7, = ax.plot(
                [i*loss_interval for i in range(len(self.his_loss[0]))],
                loss_sum, 'co-')
            ax.legend((l1, l2, l3, l4, l5, l6, l7),
                    ('dis', 'push', 'ang', 'orth', 'pad', 'ae',  'sum'))
            # loss
            plt.title('loss history')

        plt.tight_layout()
        if save:
            plt.savefig(title_+'.png', dpi=400)
        plt.close()
        

    def SaveGIF(self, path):

        gif_images_path = os.listdir(path+'/')

        gif_images_path.sort()
        print(gif_images_path)
        gif_images = []
        for _, path_ in enumerate(gif_images_path):
            print(path_)
            if '.png' in path_:
                gif_images.append(imageio.imread(path+'/'+path_))
        imageio.mimsave(path+'/'+"latent.gif", gif_images, fps=10)
