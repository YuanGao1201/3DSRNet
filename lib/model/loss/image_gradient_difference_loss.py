import torch
import torch.nn as nn
import torch.nn.functional as F


class igdl_loss(nn.Module):
    def __init__(self, igdl_p=1.0):
        super(igdl_loss, self).__init__()
        self.igdl_p = igdl_p

    def calculate_x_gradient(self, images):
        x_gradient_filter = torch.Tensor(
            [
                [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            ]
        ).cuda()
        x_gradient_filter = x_gradient_filter.view(3, 1, 3, 3)
        # result = torch.functional.F.conv2d(images, x_gradient_filter, groups=3, padding=(1, 1))
        result = F.conv2d(images, x_gradient_filter, groups=1, padding=(1, 1))
        return result

    def calculate_y_gradient(self, images):
        y_gradient_filter = torch.Tensor(
            [
                [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
            ]
        ).cuda()
        y_gradient_filter = y_gradient_filter.view(3, 1, 3, 3)
        # result = torch.functional.F.conv2d(images, y_gradient_filter, groups=3, padding=(1, 1))
        result = F.conv2d(images, y_gradient_filter, groups=1, padding=(1, 1))
        return result

    def forward(self, correct_images, generated_images):
        correct_images_gradient_x = self.calculate_x_gradient(correct_images)
        generated_images_gradient_x = self.calculate_x_gradient(generated_images)
        correct_images_gradient_y = self.calculate_y_gradient(correct_images)
        generated_images_gradient_y = self.calculate_y_gradient(generated_images)
        pairwise_p_distance = torch.nn.PairwiseDistance(p=self.igdl_p)
        distances_x_gradient = pairwise_p_distance(
            correct_images_gradient_x, generated_images_gradient_x
        )
        distances_y_gradient = pairwise_p_distance(
            correct_images_gradient_y, generated_images_gradient_y
        )
        loss_x_gradient = torch.mean(distances_x_gradient)
        loss_y_gradient = torch.mean(distances_y_gradient)
        loss = 0.5 * (loss_x_gradient + loss_y_gradient)
        return loss


class igdl_loss_3D(nn.Module):
    def __init__(self, igdl_p=1.0):
        super(igdl_loss_3D, self).__init__()
        self.igdl_p = igdl_p

    def calculate_x_gradient(self, images):
        x_gradient_filter = torch.Tensor(
            [
                [[[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                 [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                 [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]],
                [[[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                 [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                 [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]],
                [[[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                 [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                 [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]
            ]
        ).cuda()
        x_gradient_filter = x_gradient_filter.view(3, 1, 3, 3, 3)
        result = F.conv3d(images, x_gradient_filter, groups=1, padding=(1, 1, 1))
        return result

    def calculate_y_gradient(self, images):
        y_gradient_filter = torch.Tensor(
            [
                [[[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                [[0, 1, 0], [0, 0, 0], [0, -1, 0]]],
                [[[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                 [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                 [[0, 1, 0], [0, 0, 0], [0, -1, 0]]],
                [[[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                 [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                 [[0, 1, 0], [0, 0, 0], [0, -1, 0]]]
            ]
        ).cuda()
        y_gradient_filter = y_gradient_filter.view(3, 1, 3, 3, 3)
        result = F.conv3d(images, y_gradient_filter, groups=1, padding=(1, 1, 1))
        return result

    def forward(self, correct_images, generated_images):
        correct_images_gradient_x = self.calculate_x_gradient(correct_images)
        generated_images_gradient_x = self.calculate_x_gradient(generated_images)
        correct_images_gradient_y = self.calculate_y_gradient(correct_images)
        generated_images_gradient_y = self.calculate_y_gradient(generated_images)
        pairwise_p_distance = torch.nn.PairwiseDistance(p=self.igdl_p)
        distances_x_gradient = pairwise_p_distance(
            correct_images_gradient_x, generated_images_gradient_x
        )
        distances_y_gradient = pairwise_p_distance(
            correct_images_gradient_y, generated_images_gradient_y
        )
        loss_x_gradient = torch.mean(distances_x_gradient)
        loss_y_gradient = torch.mean(distances_y_gradient)
        loss = 0.5 * (loss_x_gradient + loss_y_gradient)
        return loss
