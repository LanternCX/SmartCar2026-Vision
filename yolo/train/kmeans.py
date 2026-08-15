import numpy as np
from utils import yolo_cfg
from PIL import Image 
class YOLO_Kmeans:

    def __init__(self,cfg):
        
        self.cluster_number = cfg.cluster_number
        self.filename = cfg.train_data
        self.anchor_fname = cfg.cluster_anchor
        self.target_w = cfg.width
        self.target_h = cfg.height
    def iou(self, boxes, clusters,k):  # 1 box -> k clusters
        n = boxes.shape[0]
        #k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters,self.cluster_number), axis=1)])
        print(clusters.shape)
        clusters = clusters.reshape(int(self.cluster_number/3),3,2)
        accs = []
        for cluster in clusters:
            accs.append(np.mean([np.max(self.iou(boxes, cluster,3), axis=1)]))

        return accuracy,accs

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters,self.cluster_number)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open(self.anchor_fname, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        count = 0
        for line in f:
            infos = line.split("$")
            length = len(infos)
            
            img = Image.open(infos[0])
            img_w,img_h = img.size
            scale = min(self.target_w / img_w,self.target_h / img_h)
            w_scale = self.target_w / img_w
            h_scale = self.target_h / img_h
            for i in range(1, length):
                if int(infos[i].split(",")[0]) >= int(infos[i].split(",")[2]) or int(infos[i].split(",")[1]) >= int(infos[i].split(",")[3]):
                    print(line)
                    count += 1
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                width = int(width*scale)
                height = int(height*scale)
                dataSet.append([width, height])
                if(height >= self.target_h ) or (width >= self.target_w):
                    print(img_w,img_h,infos[i].split(","))
        result = np.array(dataSet)
        f.close()
        print('box error count:%d'%count)
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        avg_acc,accs = self.avg_iou(all_boxes, result)
        print("Avg Accuracy: {:.2f}%".format(
            avg_acc * 100))
        for acc in accs:
            print(" Accuracy: {:.2f}%".format(
                acc * 100))
        


if __name__ == "__main__":
    cfg = yolo_cfg()
    kmeans = YOLO_Kmeans(cfg)
    kmeans.txt2clusters()
