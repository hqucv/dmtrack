import visdom
import visdom.server
import cv2
import torch
import copy
import numpy as np
from PIL import ImageDraw
import torchvision.transforms as transforms
import torchvision


def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)

def torch_to_numpy(a: torch.Tensor):
    return a.squeeze(0).permute(1,2,0).numpy()


class VisBase:
    def __init__(self, visdom, show_data, title):
        self.visdom = visdom
        self.show_data = show_data
        self.title = title
        self.raw_data = None
        self.opts = {}

    def update(self, data, opts={}):
        self.save_data(data, opts)

        if self.show_data:
            self.draw_data()

    def save_data(self, data, opts):
        raise NotImplementedError

    def draw_data(self):
        raise NotImplementedError

    def toggle_display(self, new_mode=None):
        if new_mode is not None:
            self.show_data = new_mode
        else:
            self.show_data = not self.show_data

        if self.show_data:
            self.draw_data()
        else:
            self.visdom.close(self.title)


class VisImage(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        data = data.float()
        self.raw_data = data
        self.opts = opts

    def draw_data(self):
        self.visdom.image(self.raw_data.clone(), opts={'title': self.title}, win=self.title)


class VisHeatmap(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        data = data.squeeze(0).flip(0)
        self.raw_data = data

    def draw_data(self):
        self.visdom.heatmap(self.raw_data.clone(),  opts={'title': self.title}, win=self.title)


class VisInfoDict(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def generate_display_text(self, data):
        display_text = ''
        for key, value in data.items():
            key = key.replace('_', ' ')
            if value is None:
                display_text += '<b>{}</b>: {}<br>'.format(key, 'None')
            elif isinstance(value, (str, int)):
                display_text += '<b>{}</b>: {}<br>'.format(key, value)
            else:
                display_text += '<b>{}</b>: {:.2f}<br>'.format(key, value)

        return display_text

    def save_data(self, data, opts):
        self.raw_data = data

    def draw_data(self):
        data = copy.deepcopy(self.raw_data)
        display_text = self.generate_display_text(data)
        self.visdom.text(display_text, opts={'title': self.title}, win=self.title)


class VisText(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        self.raw_data = data

    def draw_data(self):
        data = copy.deepcopy(self.raw_data)
        self.visdom.text(data, opts={'title': self.title}, win=self.title)


class VisLinePlot(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        self.raw_data = data

    def draw_data(self):
        if isinstance(self.raw_data, (list, tuple)):
            data_y = self.raw_data[0].clone()
            data_x = self.raw_data[1].clone()
        else:
            data_y = self.raw_data.clone()
            data_x = torch.arange(data_y.shape[0])

        self.visdom.line(data_y, data_x, opts={'title': self.title}, win=self.title)


class VisTracking(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        image = data[0]
        box = data[1]

        if box is not None and isinstance(box, list):
            box = torch.tensor(box)

        self.raw_data = [image, box]

    def draw_data(self):
        disp_image = self.raw_data[0].copy()
        box = self.raw_data[1].clone()

        if max(disp_image.shape) > 480:
            resize_factor = 480.0 / float(max(disp_image.shape))
            disp_image = cv2.resize(disp_image, None, fx=resize_factor, fy=resize_factor)
            disp_rect = box * resize_factor
        else:
            disp_rect = box

        cv2.rectangle(disp_image, (int(disp_rect[0]), int(disp_rect[1])), (int(disp_rect[0] + disp_rect[2]),
                                                                           int(disp_rect[1] + disp_rect[
                                                                               3])), (0, 255, 0), 2)
        disp_image = numpy_to_torch(disp_image).squeeze(0)
        disp_image = disp_image.float()
        self.visdom.image(disp_image, opts={'title': self.title}, win=self.title)


class VisTrackingSampling(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        image = data[0]
        box = data[1]
        gt = data[2]

        if box is not None and isinstance(box, list):
            box = torch.tensor(box)

        if gt is not None and isinstance(gt, list):
            gt = torch.tensor(gt)

        self.raw_data = [image, box, gt]

    def draw_data(self):
        disp_image = self.raw_data[0].copy()
        box = self.raw_data[1].clone()
        gt = self.raw_data[2].clone()

        if max(disp_image.shape) > 480:
            resize_factor = 480.0 / float(max(disp_image.shape))
            disp_image = cv2.resize(disp_image, None, fx=resize_factor, fy=resize_factor)
            disp_rect = box * resize_factor
            gt_rect = gt * resize_factor
        else:
            disp_rect = box
            gt_rect = gt

        # samples
        for rect in disp_rect:
            cv2.rectangle(disp_image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]),
                                                                               int(rect[1] + rect[3])), (0, 0, 255), 1)
        # ground truth
        cv2.rectangle(disp_image, (int(gt_rect[0]), int(gt_rect[1])), (int(gt_rect[0] + gt_rect[2]),
                                                                 int(gt_rect[1] + gt_rect[3])), (255, 0, 0), 1)

        disp_image = numpy_to_torch(disp_image).squeeze(0)
        disp_image = disp_image.float()
        self.visdom.image(disp_image, opts={'title': self.title}, win=self.title)


class VisTraining(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        transforms = data[0]
        reference_images = data[1]
        reference_anno = data[2]
        current_images = data[3]
        current_anno = data[4]
        pred_bbox = data[5][0][:4]

        self.raw_data = [transforms,
                         reference_images,
                         reference_anno,
                         current_images,
                         current_anno,
                         pred_bbox]

    def draw_data(self):
        transforms = self.raw_data[0]
        disp_ref_image = self.raw_data[1].clone()
        ref_gt = self.raw_data[2].clone()
        disp_cur_image = self.raw_data[3].clone()
        cur_gt = self.raw_data[4].clone()
        pred_box = self.raw_data[5].clone()

        disp_ref_image = transforms(disp_ref_image)
        disp_cur_image = transforms(disp_cur_image)

        c_ref_gt_rect = ref_gt
        c_cur_gt_rect = cur_gt
        c_pred_rect = pred_box

        # --------------------- convert to numpy type --------------------
        disp_ref_image = np.array(torchvision.transforms.ToPILImage()(disp_ref_image.cpu()))
        disp_cur_image = np.array(torchvision.transforms.ToPILImage()(disp_cur_image.cpu()))

        # ---------------------- draw rectangles -------------------------
        # for ref
        cv2.rectangle(disp_ref_image, (int(c_ref_gt_rect[0]), int(c_ref_gt_rect[1])),
                      (int(c_ref_gt_rect[2]),
                       int(c_ref_gt_rect[3])), (0, 255, 0), 1)

        # for cur
        cv2.rectangle(disp_cur_image, (int(c_cur_gt_rect[0]), int(c_cur_gt_rect[1])),
                      (int(c_cur_gt_rect[2]),
                       int(c_cur_gt_rect[3])), (0, 255, 0), 1)
        #cv2.rectangle(disp_cur_image,
        #                  (int(c_pred_rect[0]), int(c_pred_rect[1])),
        #                  (int(c_pred_rect[2]),
        #                   int(c_pred_rect[3])), (255, 0, 0), 1)

        # ----------------- convert to tensor ---------------
        disp_ref_image = numpy_to_torch(disp_ref_image).squeeze(0)
        disp_ref_image = disp_ref_image.float()
        disp_cur_image = numpy_to_torch(disp_cur_image).squeeze(0)
        disp_cur_image = disp_cur_image.float()

        # ----------------- visdom image show ------------------
        self.visdom.image(disp_ref_image, opts={'title': self.title+'_ref'}, win=self.title+'_ref')
        self.visdom.image(disp_cur_image, opts={'title': self.title+'_cur'}, win=self.title+'_cur')


class VisMotion(VisBase):
    def __init__(self, visdom, show_data, title):
        super().__init__(visdom, show_data, title)

    def save_data(self, data, opts):
        image = data
        self.raw_data = [image]

    def draw_data(self):
        disp_image = self.raw_data[0].copy()

        if max(disp_image.shape) > 480:
            resize_factor = 480.0 / float(max(disp_image.shape))
            disp_image = cv2.resize(disp_image, None, fx=resize_factor, fy=resize_factor)

        disp_image = numpy_to_torch(disp_image).squeeze(0)
        disp_image = disp_image.float()
        self.visdom.image(disp_image, opts={'title': self.title}, win=self.title)


class Visdom:
    def __init__(self, debug=0, ui_info=None, visdom_info=None):
        self.debug = debug
        self.visdom = visdom.Visdom(server=visdom_info.get('server', '127.0.0.1'), port=visdom_info.get('port', 8097))
        self.registered_blocks = {}
        self.blocks_list = []

        self.visdom.properties(self.blocks_list, opts={'title': 'Block List'}, win='block_list')
        self.visdom.register_event_handler(self.block_list_callback_handler, 'block_list')

        if ui_info is not None:
            self.visdom.register_event_handler(ui_info['handler'], ui_info['win_id'])

    def block_list_callback_handler(self, data):
        field_name = self.blocks_list[data['propertyId']]['name']

        self.registered_blocks[field_name].toggle_display(data['value'])

        self.blocks_list[data['propertyId']]['value'] = data['value']

        self.visdom.properties(self.blocks_list, opts={'title': 'Block List'}, win='block_list')

    def register(self, data, mode, debug_level=0, title='Data', opts={}):
        if title not in self.registered_blocks.keys():
            show_data = self.debug >= debug_level

            if title is not 'Tracking':
                self.blocks_list.append({'type': 'checkbox', 'name': title, 'value': show_data})

            self.visdom.properties(self.blocks_list, opts={'title': 'Block List'}, win='block_list')

            if mode == 'image':
                self.registered_blocks[title] = VisImage(self.visdom, show_data, title)
            elif mode == 'heatmap':
                self.registered_blocks[title] = VisHeatmap(self.visdom, show_data, title)
            elif mode == 'info_dict':
                self.registered_blocks[title] = VisInfoDict(self.visdom, show_data, title)
            elif mode == 'text':
                self.registered_blocks[title] = VisText(self.visdom, show_data, title)
            elif mode == 'lineplot':
                self.registered_blocks[title] = VisLinePlot(self.visdom, show_data, title)
            elif mode == 'Tracking':
                self.registered_blocks[title] = VisTracking(self.visdom, show_data, title)
            elif mode == 'TrackingSampling':
                self.registered_blocks[title] = VisTrackingSampling(self.visdom, show_data, title)
            elif mode == 'Training':
                self.registered_blocks[title] = VisTraining(self.visdom, show_data, title)
            elif mode == 'motion_visual':
                self.registered_blocks[title] = VisMotion(self.visdom, show_data, title)
            else:
                raise ValueError('Visdom Error: Unknown data mode {}'.format(mode))
        # Update
        self.registered_blocks[title].update(data, opts)
