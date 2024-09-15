import os
import torch
import numpy as np
import cv2
import math
import mediapy as mp
from torch.nn import functional as F
from kornia.geometry.conversions import axis_angle_to_rotation_matrix
import torchvision
from typing import Tuple, Literal

def bbox_to_theta(bbox):
    """
    Get theta from a bounding box.
    bbox: [x1, y1, x2, y2] in normalized space [0, 1]
    """
    ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    ct = [ct[0] * 2 - 1, ct[1] * 2 - 1]
    theta = torch.atan2(ct[1], ct[0]) / math.pi * 180 -90
    return theta

def bbox_to_phi(bbox):
    """
    Get phi from a bounding box.
    bbox: [x1, y1, x2, y2] in normalized space [0, 1]
    """
    ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    ct = [ct[0] * 2 - 1, ct[1] * 2 - 1]
    r = math.sqrt(ct[0] ** 2 + ct[1] ** 2)
    phi = - (1 - r) * 90
    return phi

def bbox_to_fov(bbox):
    """
    Get fov from a bounding box.
    bbox: [x1, y1, x2, y2] in normalized space [0, 1]
    """
    hw = [(bbox[2] - bbox[0]) / 2, (bbox[3] - bbox[1]) / 2]
    s = max(hw)
    fov = s * 360
    return fov

def bbox_to_theta_phi_fov(bbox):
    """
    Get theta, phi and fov from a bounding box.
    bbox: [x1, y1, x2, y2] in normalized space [0, 1]
    """
    ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    ct = [ct[0] * 2 - 1, ct[1] * 2 - 1]
    r = math.sqrt(ct[0] ** 2 + ct[1] ** 2)
    theta = torch.atan2(ct[1], ct[0]) / math.pi * 180 -90
    phi = - (1 - r) * 90
    hw = [(bbox[2] - bbox[0]) / 2, (bbox[3] - bbox[1]) / 2]
    s = max(hw)
    fov = s * 360
    return theta, phi, fov

def get_padxy(fisheye_shape: Tuple[int, int], delxy: Tuple[int, int], radius: int) -> Tuple[int, int]:

    h, w = fisheye_shape
    cx = w // 2 + delxy[0]
    cy = h // 2 + delxy[1]

    padxy = (cx - radius, cy - radius)

    return padxy

def center_fisheye_image(img: torch.Tensor, radius : int, delxy: Tuple) -> torch.Tensor:
    """
    Crop a fisheye image to a square image with a given radius and center.

    Args:
        img (torch.Tensor["bcwh"]): input image
        radius (int): radius of the cropped image
        delxy (Tuple): offset to center of the cropped image

    Returns:
        torch.Tensor["bcwh"]: cropped image
    """
    b, c, h, w = img.shape
    cx = w // 2 + delxy[0]
    cy = h // 2 + delxy[1]

    img = img[:, :, cy - radius:cy + radius, cx - radius:cx + radius]

    if False:
        import mediapy as mp
        out = img[:, :, cy - radius:cy + radius, cx - radius:cx + radius]
        out = out.cpu().numpy()
        out = out[0].transpose(1,2,0)
        mp.write_image("out.jpg", out)

    return img


def center_fisheye_points(pts: torch.Tensor, radius: int, padxy: Tuple = (0, 0)) -> torch.Tensor:
    """
    Crop a fisheye points to a square image with a given radius and center.

    Args:
        pts (torch.Tensor["bn"]): input points in pixels
        radius (int): radius of the cropped image
        padxy (Tuple): the top and left padding in the original image

    Returns:
        torch.Tensor["bn"]: cropped points in relative coordinates
    """
    pts[:, 0] = pts[:, 0] - padxy[0]
    pts[:, 1] = pts[:, 1] - padxy[1]

    pts[:, 0] = pts[:, 0] / (2 * radius) * 2 - 1
    pts[:, 1] = pts[:, 1] / (2 * radius) * 2 - 1

    return pts

class FisheyeMapper(torch.nn.Module):
    def __init__(self, 
                 fish_shape=(1860, 1860), 
                 equi_shape=(1000, 2000), 
                 persp_shape=(500, 500), 
                 fov=120, 
                 theta=0, 
                 phi=30, 
                 device="cuda:0"
        ):
        super(FisheyeMapper, self).__init__()

        self.fish_shape = fish_shape
        self.equi_shape = equi_shape
        self.persp_shape = persp_shape
        self.fov = fov
        self.theta = theta
        self.phi = phi
        self.device = device

        self.fish_height, self.fish_width = self.fish_shape
        self.persp_height, self.persp_width = self.persp_shape
        self.equi_height, self.equi_width = self.equi_shape

        if isinstance(fov, tuple):
            self.wFOV, self.hFOV = fov
        else:
            self.wFOV = fov
            self.hFOV = float(self.persp_height) / self.persp_width * self.fov

        self.equ_cx = (self.equi_width - 1) / 2.0
        self.equ_cy = (self.equi_height - 1) / 2.0

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))

        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
    
        self.R1 = axis_angle_to_rotation_matrix(z_axis.unsqueeze(0) * self.theta * math.pi / 180).squeeze(0).to(self.device)
        self.R2 = axis_angle_to_rotation_matrix(torch.matmul(self.R1, y_axis).unsqueeze(0) * -self.phi * math.pi / 180).squeeze(0).to(self.device)

    def equi2fish(self, x, y):

        aperture = 180 * math.pi / 180 

        # [-1, 1] -> [pi, -pi]
        lat = x * math.pi 
        lon = y * math.pi 

        # use this if we want to use equi image
        # leave for now as it is give a nicer equi image
        Px = torch.cos(lat) * torch.sin(lon)
        Py = torch.sin(lat)
        Pz = torch.cos(lat) * torch.cos(lon)
        
        phi = torch.arctan2(torch.sqrt(Px ** 2 + Pz ** 2), Py) 
        r = 2 * phi / aperture

        theta = torch.arctan2(Pz, Px)

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        return -x, y

    def fish2equi(self, x, y):
        # use this if we want to use equi image
        # leave for now as it is give a nicer equi image
        aperture = 360 * math.pi / 180
        theta = -torch.atan2(x, y)

        r = torch.sqrt(x**2 + y**2)
            
        phi = r * aperture / 2
      
        Px = torch.sin(phi) * torch.cos(theta)
        Pz = torch.cos(phi)
        Py = torch.sin(phi) * torch.sin(theta)

        lon = torch.arctan2(Py, Px)
        lat = torch.arctan2(Pz, torch.sqrt(Px ** 2 + Py ** 2))

        # [-pi, pi] -> [-1, 1]
        x = lat / math.pi
        y = lon / math.pi

        return x, y

    def persp2equi(self, y, z):

        y = y * self.w_len
        z = z * self.h_len

        x = y*0 + 1 # Way of torch.ones_like that gives static onnx graph - though kinda stupid 
        D = torch.sqrt(x**2 + y**2 + z**2)
        xyz = torch.stack((x / D, y / D, z / D),axis=-1).T

        xyz = torch.matmul(self.R1, xyz)
        xyz = torch.matmul(self.R2, xyz).T

        lat = torch.arcsin(xyz[:, 2])
        lon = torch.arctan2(xyz[:, 1] , xyz[:, 0])

        lon = lon / math.pi
        lat = -lat / math.pi * 2           

        lon = lon * self.equ_cx + self.equ_cx
        lat = lat * self.equ_cy + self.equ_cy

        return lon, lat

    def equi2persp(self, x, y):

        x = x * math.pi
        y = y * math.pi

        x_map = torch.cos(x) * torch.cos(y)
        y_map = torch.sin(x) * torch.cos(y)
        z_map = torch.sin(y)

        xyz = torch.stack((x_map,y_map,z_map),axis=-1).T

        R1 = torch.inverse(self.R1)
        R2 = torch.inverse(self.R2)

        xyz = torch.matmul(R2, xyz)
        xyz = torch.matmul(R1, xyz).T

        inverse_mask = torch.where(xyz[:,0]>0,1,0)

        xyz = xyz/xyz[:,0:1].repeat(1, 3)

        lon_map = (xyz[:,1]+self.w_len)/2/self.w_len*self.persp_width
        lat_map = (-xyz[:,2]+self.h_len)/2/self.h_len*self.persp_height

        mask = torch.where(
            (-self.w_len<xyz[:,1])
            &(xyz[:,1]<self.w_len)
            &(-self.h_len<xyz[:,2])
            &(xyz[:,2]<self.h_len),1,0)

        mask = mask * inverse_mask
        mask = mask[:, None].repeat(1,  3)

        return lon_map, lat_map, mask

    def fish2persp(self, x, y):

        x, y = self.fish2equi(x, y)

        pts_equi = torch.stack((y, x), dim=-1) * self.equi_shape[0]

        pts_equi[:, 0] = pts_equi[:, 0] + self.equi_shape[0] + 1
        pts_equi[:, 1] = pts_equi[:, 1] * 0.5 + (self.equi_shape[0] / 2) / 2
        
        pts_equi[:, 1] += 500

        pts_equi[:, 0] = pts_equi[:, 0] / self.equi_shape[0]
        pts_equi[:, 1] = pts_equi[:, 1] / self.equi_shape[1] * 2 + 0.5

        lon, lat, mask = self.equi2persp(pts_equi[:,0], pts_equi[:,1])

        pts_persp = torch.stack((lon, lat), dim=-1)

        return pts_persp[:, 0], pts_persp[:, 1], mask[:, 0]

    def persp2fish(self, x, y):

        lon, lat = self.persp2equi(x, y)

        lon = lon / (self.equi_shape[0]) + 1
        lat = (lat - 500) / (self.equi_shape[1] / 2)

        x, y = self.equi2fish(lat, lon)

        return x, y


class Persp2Equi(FisheyeMapper):
    def __init__(self, 
            fish_shape=(1860, 1860), 
            equi_shape=(1000, 2000), 
            persp_shape=(500, 500), 
            fov=120, 
            theta=0, 
            phi=30,
            device="cuda:0"
        ):
        super(Persp2Equi, self).__init__(
            fish_shape=fish_shape,
            equi_shape=equi_shape, 
            persp_shape=persp_shape, 
            fov=fov, 
            theta=theta, 
            phi=phi,
            device=device
        )
        
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, self.equi_width, device=self.device),
            torch.linspace(0.5,-0.5, self.equi_height, device=self.device),
            indexing="xy"
        )
        x = x.reshape(-1)
        y = y.reshape(-1)

        self.fish_height, self.fish_width = self.fish_shape
        self.persp_height, self.persp_width = self.persp_shape
        self.equi_height, self.equi_width = self.equi_shape
        
        lon_map, lat_map, mask = self.equi2persp(x,y)

        lon_map = lon_map.reshape([self.equi_height, self.equi_width])
        lat_map = lat_map.reshape([self.equi_height, self.equi_width])
        mask = mask.reshape([self.equi_height, self.equi_width, 3])

        grid = torch.stack((lon_map, lat_map), dim=-1).float()
        grid[..., 0] = grid[..., 0] / (self.persp_height - 1) * 2 - 1
        grid[..., 1] = grid[..., 1] / (self.persp_width - 1) * 2 - 1

        self.grid = grid.unsqueeze(0).to(self.device)
        self.mask = mask.permute(2, 0, 1).unsqueeze(0)

    def forward(self, img):

        out = F.grid_sample(img, self.grid, mode="bilinear", align_corners=True)
        out = out * self.mask

        return out


class Equi2Persp(FisheyeMapper):
    def __init__(self, 
            fish_shape=(1860, 1860), 
            equi_shape=(1000, 2000), 
            persp_shape=(500, 500), 
            fov=120, 
            theta=0, 
            phi=30,
            device="cuda:0"
        ):
        super(Equi2Persp, self).__init__(
            fish_shape=fish_shape,
            equi_shape=equi_shape, 
            persp_shape=persp_shape, 
            fov=fov, 
            theta=theta, 
            phi=phi,
            device=device
        )
        y_map = -torch.tile(torch.linspace(1, -1, self.persp_width, device=self.device), [self.persp_height,1])
        z_map = -torch.tile(torch.linspace(-1, 1, self.persp_height, device=self.device), [self.persp_width,1]).T
        
        y_map = y_map.reshape(-1)
        z_map = z_map.reshape(-1)

        lon, lat = self.persp2equi(y_map, z_map)

        lon = lon.reshape([self.persp_height, self.persp_width])
        lat = lat.reshape([self.persp_height, self.persp_width])

        grid = torch.stack((lon, lat), dim=-1).float()
        grid[..., 0] = grid[..., 0] / (self.equi_width - 1) * 2 - 1
        grid[..., 1] = grid[..., 1] / (self.equi_height - 1) * 2 - 1

        self.grid = grid.unsqueeze(0).to(self.device)

    def forward(self, img):

        out = F.grid_sample(img, self.grid, mode="bilinear", align_corners=True)
        return out


class Equi2Fish(FisheyeMapper):
    def __init__(self, 
            fish_shape=(1860, 1860), 
            equi_shape=(1000, 2000), 
            persp_shape=(500, 500), 
            fov=120, 
            theta=0, 
            phi=30,
            device="cuda:0"
        ):
        super(Equi2Fish, self).__init__(
            fish_shape=fish_shape,
            equi_shape=equi_shape, 
            persp_shape=persp_shape, 
            fov=fov, 
            theta=theta, 
            phi=phi,
            device=device
        )

        x_fish, y_fish = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.fish_shape[0], device=self.device),
            torch.linspace(-1.0, 1.0, self.fish_shape[1], device=self.device), 
            indexing='xy'
        )

        self.fish_height, self.fish_width = self.fish_shape
        self.persp_height, self.persp_width = self.persp_shape
        self.equi_height, self.equi_width = self.equi_shape

        x_fish = x_fish.reshape(-1)
        y_fish = y_fish.reshape(-1)

        mask = (x_fish ** 2 + y_fish ** 2) < 1
        x_fish_valid = x_fish[mask]
        y_fish_valid = y_fish[mask]

        x_valid, y_valid = self.fish2equi(x_fish_valid, y_fish_valid)

        x = torch.zeros_like(x_fish)
        y = torch.zeros_like(y_fish)
        x[mask] = x_valid
        y[mask] = y_valid

        grid = torch.stack([y, x], dim=-1)
        grid = grid.reshape(1, self.fish_shape[1], self.fish_shape[0], 2)
        grid[..., 0] = grid[..., 0]
        grid[..., 1] = grid[..., 1] * 2
        self.grid = grid.to(self.device)

    def forward(self, img):
        out = F.grid_sample(img, self.grid, mode="bilinear", align_corners=True)
        return out


class Fish2Equi(FisheyeMapper):
    def __init__(self, 
            fish_shape=(1860, 1860), 
            equi_shape=(1000, 2000), 
            persp_shape=(500, 500), 
            fov=120, 
            theta=0, 
            phi=30,
            device="cuda:0"
        ):
        super(Fish2Equi, self).__init__(
            fish_shape=fish_shape,
            equi_shape=equi_shape, 
            persp_shape=persp_shape, 
            fov=fov, 
            theta=theta, 
            phi=phi,
            device=device,
        )

        lat, lon = torch.meshgrid(
            torch.linspace(0, 0.5, equi_shape[0] // 2, device=self.device),
            torch.linspace(-1, 1, equi_shape[1], device=self.device), 
            indexing='xy'
        )
        lat, lon = lat.reshape(-1), lon.reshape(-1)

        x_fish, y_fish = self.equi2fish(lat, lon)

        self.fish_height, self.fish_width = self.fish_shape
        self.persp_height, self.persp_width = self.persp_shape
        self.equi_height, self.equi_width = self.equi_shape

        grid = torch.stack([x_fish, y_fish], dim=-1)
        grid = grid.reshape(1, equi_shape[1], equi_shape[0] // 2, 2).permute(0, 2, 1, 3)
        self.grid = grid.to(self.device)

    def forward(self, img):
        out = F.grid_sample(img, self.grid, mode="bilinear", align_corners=True)
        return out


class Fish2Persp(FisheyeMapper):
    def __init__(self, 
            fish_shape=(1860, 1860), 
            equi_shape=(1000, 2000), 
            persp_shape=(500, 500), 
            fov=120, 
            theta=0, 
            phi=30,
            device="cuda:0"
        ):
        super(Fish2Persp, self).__init__(
            fish_shape=fish_shape,
            equi_shape=equi_shape, 
            persp_shape=persp_shape, 
            fov=fov, 
            theta=theta, 
            phi=phi,
            device=device,
        )
        y_map = -torch.tile(torch.linspace(1, -1, self.persp_width, device=self.device), [self.persp_height,1])
        z_map = -torch.tile(torch.linspace(-1, 1, self.persp_height, device=self.device), [self.persp_width,1]).T
                
        y_map = y_map.reshape(-1)
        z_map = z_map.reshape(-1)

        self.fish_height, self.fish_width = self.fish_shape
        self.persp_height, self.persp_width = self.persp_shape
        self.equi_height, self.equi_width = self.equi_shape

        x_fish, y_fish = self.persp2fish(y_map, z_map)

        grid = torch.stack([x_fish, y_fish], dim=-1)
        grid = grid.reshape(1, self.persp_width, self.persp_height, 2)
        self.grid = grid.to(self.device)

    def forward(self, img, mode="bilinear"):
        out = F.grid_sample(img, self.grid, mode=mode, align_corners=True)
        return out


class Persp2Fish(FisheyeMapper):
    def __init__(self, 
            fish_shape=(1860, 1860), 
            equi_shape=(1000, 2000), 
            persp_shape=(500, 500), 
            fov=120, 
            theta=0, 
            phi=30,
            device="cuda:0"
        ):
        super(Persp2Fish, self).__init__(
            fish_shape=fish_shape,
            equi_shape=equi_shape, 
            persp_shape=persp_shape, 
            fov=fov, 
            theta=theta, 
            phi=phi,
            device=device,
        )
               
        x_fish, y_fish = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.fish_shape[0], device=self.device),
            torch.linspace(-1.0, 1.0, self.fish_shape[1], device=self.device), 
            indexing='xy'
        )

        self.fish_height, self.fish_width = self.fish_shape
        self.persp_height, self.persp_width = self.persp_shape
        self.equi_height, self.equi_width = self.equi_shape
    
        x_fish = x_fish.reshape(-1)
        y_fish = y_fish.reshape(-1)

        mask_input = (x_fish ** 2 + y_fish ** 2) < 1
        x_fish_valid = x_fish[mask_input]
        y_fish_valid = y_fish[mask_input]

        x_valid, y_valid, mask_valid = self.fish2persp(x_fish_valid, y_fish_valid)

        x = torch.zeros_like(x_fish)
        y = torch.zeros_like(y_fish)
        mask = torch.zeros_like(x_fish)
        x[mask_input] = x_valid
        y[mask_input] = y_valid
        mask[mask_input] = mask_valid.float()

        grid = torch.stack([x, y], dim=-1)
        grid = grid.reshape(1, self.fish_shape[1], self.fish_shape[0], 2)
        grid[..., 0] = grid[..., 0] / (self.persp_height - 1) * 2 - 1
        grid[..., 1] = grid[..., 1] / (self.persp_width - 1) * 2 - 1

        self.grid = grid.to(self.device)

        mask = mask.reshape(1, self.fish_shape[1], self.fish_shape[0])
        self.mask = mask.permute(2, 0, 1).unsqueeze(0)

    def forward(self, img):
        out = F.grid_sample(img, self.grid, mode="bilinear", align_corners=True)
        return out
    

class Fish2NPersp(torch.nn.Module):
    def __init__(
            self,
            fish_shape=(1860, 1860), 
            equi_shape=(1000, 2000), 
            persp_shape=(512, 512), 
            fov=(120, 120, 120, 120), 
            theta=(0, 90, 180, 270), 
            phi=(-60, -60, -60, -60),
            device="cuda:0"
        ):
        super().__init__()
        assert len(fov) == len(theta) == len(phi), "fov, theta and phi must have the same length"

        self.n_views = len(fov)

        self.pts_mappers = torch.nn.ModuleList()
        grids = []
        for i in range(self.n_views):
            f2p = Fish2Persp(
                fish_shape=fish_shape, 
                equi_shape=equi_shape, 
                persp_shape=persp_shape, 
                fov=fov[i], 
                theta=theta[i], 
                phi=phi[i], 
                device=device
            )

            grids.append(f2p.grid)
            self.pts_mappers.append(
                FisheyeMapper(
                    fish_shape=fish_shape, 
                    equi_shape=equi_shape, 
                    persp_shape=persp_shape, 
                    fov=fov[i], 
                    theta=theta[i], 
                    phi=phi[i], 
                    device=device
                )
            )

        grids = torch.cat(grids, dim=0).to(self.pts_mappers[0].device)
        self.register_buffer("grids", grids)
        self.register_buffer("R_C2W", self.get_rotation())

    def persp2fish(self, pts, i=0):
        h, w = self.pts_mappers[i].persp_shape

        x = pts[:, 0] / h * 2 - 1
        y = -pts[:, 1] / w * 2 + 1 
    
        x, y = self.pts_mappers[i].persp2fish(x, y)
    
        fish_h, fish_w = self.pts_mappers[i].fish_shape
        pts = torch.stack((x, y), dim=-1) * fish_h / 2 + fish_h / 2

        return pts

    def fish2persp(self, pts, i=0):
        x, y, mask = self.pts_mappers[i].fish2persp(pts[:,0], pts[:,1])
        pts = torch.stack((x, y), dim=0)
        pts = torch.stack((pts[0], pts[1]), dim=-1)
        return pts, mask
    
    def forward(self, img, mode="bilinear"):
        out = []
        for i in range(self.grids.shape[0]):
            out.append(F.grid_sample(
                img, 
                self.grids[i].unsqueeze(0).expand(img.shape[0], -1, -1, -1),
                mode=mode, 
                align_corners=True
            ))
        
        return torch.cat(out, dim=0)
    
    def get_rotation(self):
        """
           Define rotation matrices from the fish-eye camera to the perspective camera
        """
        R_C2W = []
        for i in range(self.n_views):
            if i == 0:
                gamma = 0
            elif i == 1:
                gamma = 180
            elif i == 2:
                gamma = 90
            elif i == 3:
                gamma = 270

            phi = 0
            theta = 30

            gamma = gamma / 180 * math.pi
            phi = phi / 180 * math.pi
            theta = theta / 180 * math.pi
            Rx = torch.tensor([
                [1, 0, 0], 
                [0, math.cos(theta), -math.sin(theta)], 
                [0, math.sin(theta), math.cos(theta)],
            ])

            Ry = torch.tensor([
                [math.cos(phi), 0, math.sin(phi)], 
                [0, 1, 0], 
                [-math.sin(phi), 0, math.cos(phi)],
            ])

            Rz = torch.tensor([
                [math.cos(gamma), -math.sin(gamma), 0], 
                [math.sin(gamma), math.cos(gamma), 0], 
                [0, 0, 1],
            ])

            R_W2C = Rz @ Ry @ Rx
            R_C2W.append(R_W2C.T)

        return torch.stack(R_C2W, dim=0)
    

class NPersp2Fish(torch.nn.Module):
    def __init__(
            self,
            persp_shape=(512, 512), 
            fov=(120, 120, 120, 120), 
            theta=(0, 90, 180, 270), 
            phi=(-60, -60, -60, -60),
        ):
        super().__init__()

        self.n_views = len(fov)

        self.grids = []
        for i in range(self.n_views):
            p2f = Persp2Fish(persp_shape=persp_shape, fov=fov[i], theta=theta[i], phi=phi[i])
            self.grids.append(p2f.grid)

        self.grids = torch.cat(self.grids, dim=0)

    def forward(self, img):
        out = F.grid_sample(
            img.expand(self.n_views, -1, -1, -1), 
            self.grids, 
            mode="bilinear", 
            align_corners=True
        )

        return out

    
class NPersp2Fish(Fish2NPersp):
    def __init__(
            self,
            fish_shape=(960, 960), 
            equi_shape=(1000, 2000), 
            persp_shape=(512, 512), 
            fov=(120, 120, 120, 120), 
            theta=(0, 90, 180, 270), 
            phi=(-60, -60, -60, -60),
            grid_sample_downscale_factor=8,
            device="cuda:0"
        ):
        super().__init__(
            fish_shape=fish_shape,
            equi_shape=equi_shape, 
            persp_shape=persp_shape, 
            fov=fov, 
            theta=theta, 
            phi=phi,
            device=device,
        )

        self.n_views = len(fov)
        self.fish_shape = fish_shape
        self.grid_sample_downscale_factor = grid_sample_downscale_factor
        self.grid_sample_shape = (fish_shape[0] // grid_sample_downscale_factor, fish_shape[1] // grid_sample_downscale_factor)
        self.persp_height, self.persp_width = persp_shape
        self.device = device

        self.fish_height, self.fish_width = self.fish_shape

        grids = []
        masks = []
        self.mask_indices = []
        self.x_fish, self.y_fish = [], []
        weights = []
        for i in range(len(fov)):
            x_fish, y_fish = torch.meshgrid(
                torch.linspace(-1.0, 1.0, self.grid_sample_shape[0], device=self.device),
                torch.linspace(-1.0, 1.0, self.grid_sample_shape[1], device=self.device), 
                indexing='xy'
            )

            x_fish = x_fish.reshape(-1)
            y_fish = y_fish.reshape(-1)

            mask_input = (x_fish ** 2 + y_fish ** 2) < 1
            x_fish_valid = x_fish[mask_input]
            y_fish_valid = y_fish[mask_input]

            pts = torch.stack((x_fish_valid, y_fish_valid), dim=-1)
            pts_valid, mask_valid = self.fish2persp(pts, i=i)
            x_valid, y_valid = pts_valid[:, 0], pts_valid[:, 1]

            x = torch.zeros_like(x_fish)
            y = torch.zeros_like(y_fish)
            mask = torch.zeros_like(x_fish)
            x[mask_input] = x_valid
            y[mask_input] = y_valid
            mask[mask_input] = mask_valid.float()

            grid = torch.stack([x, y], dim=-1)
            grid = grid.reshape(1, self.grid_sample_shape[1], self.grid_sample_shape[0], 2)
            grid[..., 0] = grid[..., 0] / (self.persp_height - 1) * 2 - 1
            grid[..., 1] = grid[..., 1] / (self.persp_width - 1) * 2 - 1
            grid.clip_(-1.1, 1.1)
            grids.append(grid)

            mask = mask.reshape(1, 1, self.grid_sample_shape[1], self.grid_sample_shape[0])
            masks.append(mask)

            mask = mask[0,0].bool()
            mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 960, 960]
            mask_expanded = mask_expanded.expand(-1, 8, -1, -1)  # [1, 8, 960, 960]

            # Flatten the mask and get the indices where it is True
            indices = mask_expanded.flatten().nonzero().squeeze(1)
            self.mask_indices.append(torch.from_numpy(indices.cpu().numpy()))

            custom_grid = (grid + 1) / 2

            x_fish  = custom_grid[0, :, :, 0][mask]
            y_fish  = custom_grid[0, :, :, 1][mask]
            self.x_fish.append(x_fish)
            self.y_fish.append(y_fish)

            x = torch.linspace(-1, 1, steps=self.persp_width).unsqueeze(0).repeat(self.persp_height, 1)
            y = torch.linspace(-1, 1, steps=self.persp_height).unsqueeze(1).repeat(1, self.persp_width)

            distance_from_center = torch.sqrt(x**2 + y**2)
            weigth = (1 - distance_from_center.clamp_(0, 0.99)) ** 2

            weights.append(weigth)

        self.grids = torch.cat(grids, dim=0)
        self.masks = torch.cat(masks, dim=0)
        self.weights = F.grid_sample(
            torch.stack(weights, dim=0)[:, None].to(self.device), 
            self.grids, 
            mode="bilinear", 
            align_corners=True
        )
 
    def to_gpu(self):
        self.grids = self.grids.cuda()
        self.masks = self.masks.cuda()
        self.weights = self.weights.cuda()

    def forward(self, img, reduction="mean"):
        batch_size = img.shape[0] // self.n_views
        cam_id = torch.arange(img.shape[0]) //batch_size
        out = F.grid_sample(
            img, 
            self.grids[cam_id],
            mode="bilinear", 
            align_corners=True
        )
        out = out.reshape(self.n_views, batch_size, int(out.shape[1]), int(out.shape[2]), int(out.shape[3])).permute(1, 0, 2, 3, 4)

        if False:
            import mediapy as mp
            for i in range(self.n_views):
                mp.write_image(f"out{i}.png", out[i, 0].cpu().numpy() / 2**16)

        out = out * self.masks[None]
        if reduction == "mean":
            num = self.masks.sum(dim=0)
            out = out.sum(dim=1) / (num + 1e-6)
            out = out * (num > 0).float()
        elif reduction == "max":
            out = torch.max(out, dim=1)[0]
        elif reduction == "sum":
            out = out.sum(dim=1)
        elif reduction == "median":
            out = torch.median(out, dim=1)[0]
        elif reduction == "weighted_mean":
            out = out * self.weights
            num = self.weights.sum(dim=0)
            out = out.sum(dim=1) / (num + 1e-6)
            out = out * (num > 0).float()
        elif reduction == "weighted":
            out = out * self.weights
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"reduction {reduction} not supported")

        if False:
            mp.write_image("avg.png", out.cpu().permute(1,2,0).numpy().astype(np.uint8))

        if out.ndim == 3:
            out = out[None]

        if reduction == "none" or reduction == "weighted":
            out = out.reshape(self.n_views * batch_size, int(out.shape[2]), int(out.shape[3]), int(out.shape[4]))

        if self.fish_shape[0] % self.grid_sample_downscale_factor != 0 or\
            self.fish_shape[1] % self.grid_sample_downscale_factor != 0:
            return F.interpolate(out, size=self.fish_shape, mode="bilinear")

        return F.interpolate(out, scale_factor=self.grid_sample_downscale_factor, mode="bilinear")

if __name__ == "__main__":

    img = cv2.imread("/root/autodl-tmp/fisheye-3d-challenge/data/frames/output_0001.png", cv2.IMREAD_COLOR)
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)

    equi_shape = (1000, 2000)
    theta = 0
    phi = -60

    os.makedirs("equi", exist_ok=True)
    # persp_shape = (500, 500)

    # crop fisheye image with calibration parameters
    delx = 0
    dely = 0
    radius = img.shape[2] // 2

    assert img.shape[2] == img.shape[3], f"image is not square {img.shape}"

    x = torch.linspace(0, 1000, 1000)
    y = torch.ones_like(x) * 500
    pts_fish = torch.stack((x, y), dim=-1)
    y = torch.linspace(0, 1000, 1000)
    x = torch.ones_like(y) * 500
    pts_fish = torch.cat((pts_fish, torch.stack((x, y), dim=-1)), dim=0)
    y = torch.linspace(0, 1000, 1000)
    x = torch.ones_like(y) * 250
    pts_fish = torch.cat((pts_fish, torch.stack((x, y), dim=-1)), dim=0)
    x = torch.linspace(0, 1000, 1000)
    y = torch.ones_like(x) * 250
    pts_fish = torch.cat((pts_fish, torch.stack((x, y), dim=-1)), dim=0)
    x = torch.linspace(0, 1000, 1000)
    y = torch.ones_like(x) * 750
    pts_fish = torch.cat((pts_fish, torch.stack((x, y), dim=-1)), dim=0)
    y = torch.linspace(0, 1000, 1000)
    x = torch.ones_like(y) * 750
    pts_fish = torch.cat((pts_fish, torch.stack((x, y), dim=-1)), dim=0)

    def get_color(i):
        if i < 1000:
            color = (0, 255, 0)
        elif i < 2000:
            color = (0, 0, 255)
        elif i < 3000:
            color = (255, 0, 0)
        elif i < 4000:
            color = (0, 255, 255)
        elif i < 5000:
            color = (255, 0, 255)
        elif i < 6000:
            color = (255, 255, 0)
        elif i < 7000:
            color = (255, 255, 255)
        elif i < 8000:
            color = (0, 0, 0)
        return color

    def save_image(name, img, pts):
        pts = pts.cpu().numpy().astype(np.int32)
        out_copy = (img.cpu().squeeze(0).permute(1,2,0).numpy()).astype(np.uint8).copy()
        for i in range(len(pts)):
            color = get_color(i)
            cv2.circle(out_copy, (pts[i][0], pts[i][1]), 10, color, -1)
        mp.write_image(name, out_copy)

    
    pts_fish[:, 0] *= img.shape[2] / equi_shape[0]
    pts_fish[:, 1] *= img.shape[3] / (equi_shape[1] / 2)

    save_image("equi/fish_in.png", img, pts_fish)

    # map fish 2 equi
    fish2equi = Fish2Equi()
    equi = fish2equi(img.cuda())

    pts_fish[:, 0] /= img.shape[2]
    pts_fish[:, 1] /= img.shape[3]
    pts_fish = pts_fish * 2 - 1
    pts_fish_clone = pts_fish.clone()
    x, y = FisheyeMapper().fish2equi(pts_fish[:,0], pts_fish[:,1])

    pts_equi = torch.stack((y, x), dim=-1) * equi_shape[0]

    pts_equi[:, 0] = pts_equi[:, 0] + equi_shape[0]
    pts_equi[:, 1] = pts_equi[:, 1] * 0.5 + (equi_shape[0] / 2) / 2

    save_image("equi/fish2equi.png", equi.cpu(), pts_equi)

    tmp = torch.zeros((1, 3, 1000, 2000))
    tmp[:, :, 500:, :] = equi
    equi = tmp

    pts_equi[:, 1] += 500

    save_image("equi/fish2equi_padded.png", equi.cpu(), pts_equi)
    
    # map equi 2 persp
    pts_equi_clone = pts_equi.clone()
    
    equi2persp = Equi2Persp(theta=theta, phi=phi, fov=120)
    persp = equi2persp(equi.cuda())

    pts_equi_clone[:, 0] = pts_equi_clone[:, 0] / equi_shape[0]
    pts_equi_clone[:, 1] = pts_equi_clone[:, 1] / equi_shape[1] * 2 + 0.5

    pts_equi_clone = pts_equi_clone.cuda()
    lon, lat, mask = FisheyeMapper(theta=theta, phi=phi, fov=120).equi2persp(pts_equi_clone[:,0], pts_equi_clone[:,1])
    pts_persp = torch.stack((lon, lat), dim=-1)

    pts_persp = pts_persp * mask[:,:2] 
    save_image(f"equi/equi2persp.png", persp, pts_persp)

    # map persp 2 equi
    persp2equi = Persp2Equi(theta=theta, phi=phi)
    equi_remap = persp2equi(persp.cuda())

    pts_persp[:, 0] = pts_persp[:, 0] / 500 * 2 - 1
    pts_persp[:, 1] = -pts_persp[:, 1] / 500 * 2 + 1 
    lon, lat = FisheyeMapper(theta=theta, phi=phi).persp2equi(pts_persp[:,0], pts_persp[:,1])
    pts_equi_remap = torch.stack((lon, lat), dim=-1)

    save_image("equi/persp2equi.png", equi_remap, pts_equi_remap)

    #TODO: (fred) do not resize here.
    equi_remap = equi_remap[:, :, 500:, :]
    pts_equi_remap[:, 1] -= 500

    # map equi 2 fish
    equi2fish = Equi2Fish(theta=theta, phi=phi)
    fish_remap = equi2fish(equi_remap)

    pts_equi_remap[:, 0] = pts_equi_remap[:, 0] / (equi_remap.shape[2] * 2) + 1
    pts_equi_remap[:, 1] = pts_equi_remap[:, 1] / (equi_remap.shape[3] / 2)

    x, y = FisheyeMapper(theta=theta, phi=phi).equi2fish(pts_equi_remap[:,1], pts_equi_remap[:,0])
    pts_fish_remap = torch.stack((x, y), dim=-1) * fish_remap.shape[2] / 2 + fish_remap.shape[2] / 2
    
    save_image("equi/equi2fish.png", fish_remap, pts_fish_remap)

    ## in one step ##

    # map fish 2 persp
    fish2persp = Fish2Persp(theta=theta, phi=phi)
    persp_direct = fish2persp(img.cuda())

    pts_fish_clone = pts_fish_clone.cuda()
    pts_persp_direct = FisheyeMapper(theta=theta, phi=phi).fish2persp(pts_fish_clone[:,0], pts_fish_clone[:,1])
    pts_persp_direct = torch.stack((pts_persp_direct[0], pts_persp_direct[1]), dim=-1)
    save_image("equi/fish2persp.png", persp_direct, pts_persp_direct)

    # map persp 2 fish
    persp2fish = Persp2Fish(theta=theta, phi=phi)
    fish_remap = persp2fish(persp_direct)

    pts_persp_direct[:, 0] = pts_persp_direct[:, 0] / 500 * 2 - 1
    pts_persp_direct[:, 1] = -pts_persp_direct[:, 1] / 500 * 2 + 1 
    x, y = FisheyeMapper(theta=theta, phi=phi).persp2fish(pts_persp_direct[:,0], pts_persp_direct[:,1])
    pts_fish_direct = torch.stack((x, y), dim=-1) * fish_remap.shape[2] / 2 + fish_remap.shape[2] / 2
    save_image("equi/persp2fish.png", fish_remap, pts_fish_direct)

    # map fish 2 four persp
    fish2npersp = Fish2NPersp()
    four_persp_direct = fish2npersp(img.cuda())
    four_persp = torchvision.utils.make_grid(four_persp_direct.cpu(), nrow=2, padding=5).permute(1,2,0).numpy().astype(np.uint8)
    mp.write_image("equi/four_persp.png", four_persp)

    for i in range(4):
        pts_persp_direct, mask = fish2npersp.fish2persp(pts_fish_clone, i=i)
        save_image(f"equi/fish2persp_{i}.png", four_persp_direct[i:i+1], pts_persp_direct)
    
    # map four persp 2 fish
    npersp2fish = NPersp2Fish(grid_sample_downscale_factor=1)
    fish_remap = npersp2fish(four_persp_direct)
    mp.write_image("equi/four_fish.png", fish_remap.cpu().squeeze(0).permute(1,2,0).numpy().astype(np.uint8))

    images = []
    for theta in range(360):

        fish2persp = Fish2Persp(theta=theta, phi=-60)
        persp_direct = fish2persp(img.cuda())

        x, y, mask = fish2persp.fish2persp(pts_fish_clone[:,0], pts_fish_clone[:,1])
        pts_persp_direct = torch.stack((x, y), dim=-1)

        os.makedirs("rot", exist_ok=True)
        save_image(f"rot/{i}.png", persp_direct, pts_persp_direct)

        images.append(persp_direct.cpu().squeeze(0).permute(1,2,0).numpy().astype(np.uint8))

    mp.write_video("rot.mp4", images, fps=10)

    # map fish 2 bbox
    ct = [0.55, 0.29]
    size = [0.25, 0.25]
    bbox = torch.tensor([ct[0] - size[0] / 2, ct[1] - size[1] / 2, ct[0] + size[0] / 2, ct[1] + size[1] / 2]).float().unsqueeze(0)

    theta, phi, fov = bbox_to_theta_phi_fov(bbox[0])

    fish_img_bbox = img[0].permute(1,2,0).numpy().copy().astype(np.uint8)
    bbox[:, [0, 1, 2, 3]] *= torch.tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2]])[None, :]
    cv2.rectangle(fish_img_bbox, (int(bbox[0, 0]), int(bbox[0, 1])), (int(bbox[0, 2]), int(bbox[0, 3])), (0, 255, 0), 2)
    mp.write_image("equi/fish_in_bbox.png", fish_img_bbox)

    fish2persp = Fish2Persp(fov=fov, theta=theta, phi=phi)
    img_bbox = fish2persp(img.cuda())

    mp.write_image("equi/fish_out.png", img_bbox.cpu()[0].permute(1,2,0).numpy().astype(np.uint8))