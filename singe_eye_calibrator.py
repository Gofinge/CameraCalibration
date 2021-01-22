import cv2
import numpy as np


class SingleEyeCalibrator:
    def __init__(self, src="calibrationImages/1.bmp"):
        # Step 1: Capture Image
        self.image = cv2.imread(src)
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Step 2: Detect Main Points
        self.points = self.simple_circle_detect(self.grey)
        self.mainPoints, self.subPoints = self.points_cluster()
        self.maxSize = max([point.size for point in self.mainPoints])
        image = self.image.copy()
        image = self.plot_point(image, self.mainPoints, with_order=True)
        # image = self.plot_point(image, self.subPoints, color=[0, 0, 255])
        self.image_show(image)
        # Step 3: Correct Main Points, Mark info
        self.size = (5, 6)  # input by operator
        self.zeroPoint = ["G", 2]
        # TODO: correct main points
        # Step 4: Store Main Points to matrix
        self.mainPointsMatrix = self.generate_points_matrix(self.mainPoints, self.size, 20)
        # Step 5: Mark up-left Main Point
        # self.zeroPoint = self.mark_zero_point()

        # Step 5: Single Camera Calibration
        # TODO: Single Camera Calibration with main point
        # Step 6: Grid Base SubPoints Search

        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

        for i in range(self.size[0] - 1):
            for j in range(self.size[1] - 1):
                block_grey, block, dist_between_blob = self.sub_points_roi([i, j])
                points = self.sub_points_detect(block_grey)
                block = self.plot_point(block, points, with_order=True, color=(0, 0, 255))
                self.image_show(block, name="block({}, {})".format(i, j))
                # TODO: correct subPoint
                sub_point_matrix = self.generate_points_matrix(points, (11, 11), 2)
                zero_point = sub_point_matrix[0][0]["pt"]
                for h in range(11):
                    for w in range(11):
                        sub_point_matrix[h][w]["pt"] = \
                            [sub_point_matrix[h][w]["pt"][0] - zero_point[0] + self.mainPointsMatrix[i][j]["pt"][0],
                             sub_point_matrix[h][w]["pt"][1] - zero_point[1] + self.mainPointsMatrix[i][j]["pt"][1]]
                self.mainPointsMatrix[i][j]["subPointMatrix"] = sub_point_matrix

        # Step 7: Point localization
        sample_point = (1000, 1000)
        image = self.image.copy()
        image = cv2.circle(image, sample_point, 5, (0, 0, 255))
        self.image_show(image)
        block_reg, block_idx = self.locate_block(sample_point, self.mainPointsMatrix, self.size)
        point_reg, point_idx = self.locate_block(sample_point,
                                                 self.mainPointsMatrix[block_idx[0]][block_idx[1]]["subPointMatrix"],
                                                 [11, 11])
        # mm
        dis = self.locate_pixel(sample_point, block_idx, point_idx)
        print("pause")

    def locate_pixel(self, pt, main_block_idx, sub_block_idx):
        main_block = self.mainPointsMatrix[main_block_idx[0]][main_block_idx[1]]["subPointMatrix"]
        keypoint = main_block[sub_block_idx[0]][sub_block_idx[1]]["pt"]
        keypoint_right = main_block[sub_block_idx[0]][sub_block_idx[1] + 1]["pt"]
        keypoint_down = main_block[sub_block_idx[0] + 1][sub_block_idx[1]]["pt"]
        density = main_block[sub_block_idx[0]][sub_block_idx[1]]["density"]
        dis = [self.cal_distance(pt, keypoint) * self.cal_distance(pt, keypoint_right) /
               self.cal_distance(keypoint, keypoint_right) * density[0],
               self.cal_distance(pt, keypoint) * self.cal_distance(pt, keypoint_down) /
               self.cal_distance(keypoint, keypoint_down) * density[1]
               ]
        return dis

    def locate_block(self, pt, pointMatrix, size):
        dis_matrix = [[0 for _ in range(size[1])] for _ in range(size[0])]
        idx = [-1, -1]
        min_dist = np.inf
        reg = True
        for i in range(size[0] - 1):
            for j in range(size[1] - 1):
                main_point = pointMatrix[i][j]
                dis = self.cal_distance(main_point["pt"], pt)
                dis_matrix[i][j] = dis
                if pt[0] > main_point["pt"][0] and pt[1] > main_point["pt"][1]:
                    if dis < min_dist:
                        min_dist = dis
                        idx = [i, j]

        # target point is not in a block, just return the closest main point
        if idx == [-1, -1]:
            for i in range(self.size[0] - 1):
                for j in range(self.size[1] - 1):
                    if dis_matrix[i][j] < min_dist:
                        min_dist = dis_matrix[i][j]
                        idx = [i, j]
                        reg = False

        return reg, idx

    @staticmethod
    def cal_distance(pta, ptb):
        return ((pta[0] - ptb[0]) ** 2 + (pta[1] - ptb[1]) ** 2) ** 0.5

    def mark_zero_point(self):
        zero_point = self.mainPointsMatrix[0][0]
        self.image_show(self.image[int(zero_point[1] - self.maxSize * 5): int(zero_point[1] + self.maxSize * 5),
                        int(zero_point[0] - self.maxSize * 5): int(zero_point[0] + self.maxSize * 5)])
        # TODO: input id
        name = None
        return name

    def sub_points_roi(self, idx):
        up_left = self.mainPointsMatrix[idx[0]][idx[1]]["pt"]
        up_right = self.mainPointsMatrix[idx[0]][idx[1] + 1]["pt"]
        down_left = self.mainPointsMatrix[idx[0] + 1][idx[1]]["pt"]
        down_right = self.mainPointsMatrix[idx[0] + 1][idx[1] + 1]["pt"]
        up = int(min(up_left[1], up_right[1]) - self.maxSize * 0.6)
        down = int(max(down_left[1], down_right[1]) + self.maxSize * 0.6) + 1
        left = int(min(up_left[0], down_left[0]) - self.maxSize * 0.6)
        right = int(max(up_right[0], down_right[0]) + self.maxSize * 0.6) + 1
        dist_between_blob = (max(down_left[1], down_right[1]) - min(up_left[1], up_right[1]) +
                             max(up_right[0], down_right[0]) - min(up_left[0], down_left[0])) / 20
        roi_grey = self.grey[up: down, left: right].copy()
        roi = self.image[up: down, left: right].copy()
        return roi_grey, roi, dist_between_blob

    def generate_points_matrix(self, points, size, unit):
        assert len(points) == size[0] * size[1], "the number of points does not fit the matrix"
        # diag order sort
        points.sort(key=lambda x: (x.pt[1] + x.pt[0] * 1.1))
        points_matrix = [[{} for _ in range(size[1])] for _ in range(size[0])]

        idx = 0
        for k in range(size[0] + size[1] - 1):
            for j in range(k + 1):
                i = k - j
                if 0 <= i < size[0] and j < size[1]:
                    points_matrix[i][j] = {"pt": points[idx].pt}
                    idx += 1

        for i in range(size[0] - 1):
            for j in range(size[1] - 1):
                pt = points_matrix[i][j]["pt"]
                pt_right = points_matrix[i][j+1]["pt"]
                pt_down = points_matrix[i+1][j]["pt"]
                density = [unit / self.cal_distance(pt, pt_down), unit / self.cal_distance(pt, pt_right)]
                points_matrix[i][j]["density"] = density
        return points_matrix

    @staticmethod
    def sub_points_detect(src):
        params = cv2.SimpleBlobDetector_Params()
        detector = cv2.SimpleBlobDetector_create(params)
        points = detector.detect(src)
        points.sort(key=lambda x: x.size, reverse=True)
        # TODO: add assert for detect point less than 121
        points = points[0: 121]
        # TODO: add sub pixel detect
        return points

    @staticmethod
    def simple_circle_detect(src, min_circularity=0.8):
        # apply cv2.SimpleBlobDetector to search circle
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = min_circularity
        detector = cv2.SimpleBlobDetector.create(params)
        points = detector.detect(src)
        return points

    def points_cluster(self):
        point_size_list = [point.size for point in self.points]
        cls = self.k_means_1d(point_size_list, [min(point_size_list), max(point_size_list)])
        flag = int(cls["cluster_centers"][1] > cls["cluster_centers"][0])
        return [self.points[i] for i in range(len(cls["labels"])) if cls["labels"][i] == flag], \
               [self.points[i] for i in range(len(cls["labels"])) if cls["labels"][i] != flag]

    @staticmethod
    def image_show(src, name="image"):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def plot_point(src, points, color=(0, 255, 0), with_order=False):
        image = cv2.drawKeypoints(src, points, src, color=color,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        if with_order:
            points.sort(key=lambda x: (x.pt[1] + x.pt[0] * 1.1))
            for i in range(len(points) - 1):

                cv2.line(image, (int(points[i].pt[0]), int(points[i].pt[1])),
                         (int(points[i + 1].pt[0]), int(points[i + 1].pt[1])), color, 1, 4)
        return image

    @staticmethod
    def k_means_1d(dataSet: list, init_centers: list):
        num_samples = len(dataSet)
        dataSet = np.array(dataSet)
        # first column stores which cluster this sample belongs to,
        # second column stores the error between this sample and its centroid
        k = len(init_centers)
        centers = init_centers
        cluster_assment = np.mat(np.zeros((num_samples, 2)))
        cluster_changed = True

        while cluster_changed:
            cluster_changed = False
            for i in range(num_samples):
                min_dist = np.inf
                min_idx = 0
                for j in range(k):
                    distance = abs(centers[j] - dataSet[i])
                    if distance < min_dist:
                        min_dist = distance
                        min_idx = j

                if cluster_assment[i, 0] != min_idx:
                    cluster_changed = True
                    cluster_assment[i, :] = min_idx, min_dist

            for j in range(k):
                points_in_cluster = dataSet[np.nonzero(cluster_assment[:, 0].A == j)[0]]
                centers[j] = np.mean(points_in_cluster, axis=0)
        return {"cluster_centers": centers,
                "labels": [cluster_assment.tolist()[i][0] for i in range(num_samples)]}




if __name__ == '__main__':
    cal = SingleEyeCalibrator()
    print("pause")
