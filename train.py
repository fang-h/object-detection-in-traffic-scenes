import paddle
import paddle.fluid as fluid
import numpy as np
from tqdm import tqdm as tqdm
# backbone
from backbone.DarkNet53 import DarkNet53
from backbone.CSPDarkNet53 import CSPDarkNet53
# model 
from model.yolov3 import YoloV3
from model.yolov4 import YoloV4
# anchor
from utils.anchors import YoloAnchors
# reader
from utils.reader import TrainDataReader
# loss
from loss_and_metrics.yolov3_loss import YoloV3Loss
# config
import config


yolov3_path = '/home/aistudio/work/code/logs/yolov3_0.pdparams'


def yolov3_train():
    train_logs = open("/home/aistudio/work/code/logs/yolov3.csv", 'w')
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        model = YoloV3()
        model_dict, _ = fluid.dygraph.load_dygraph(yolov3_path)
        model.load_dict(model_dict)
        data_reader = TrainDataReader(image_size=(640, 1152))
        # print(data_reader.get_lengths()) = 37144
        default_anchors = YoloAnchors().get_anchors()
        yolov3_loss = YoloV3Loss(config)
        lr_boundaries = [x - 2653 for x in [3e3, 6e3, 9e3, 1.1e4, 1.4e4, 1.1e5, 1.6e5, 1.8e5, 2.1e5, 2.2e5, 2.3e5]]
        lr_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-5]
        optimizer = fluid.optimizer.MomentumOptimizer(momentum=0.9,
                                                      learning_rate=fluid.layers.piecewise_decay(lr_boundaries, lr_values),
                                                      regularization=fluid.regularizer.L2Decay(5e-4),
                                                      parameter_list=model.parameters())
        # model.eval()
        for epoch in range(1, config.max_epochs):
            print(optimizer.current_step_lr())
            data_reader.shuffle()
            batch_process = tqdm(range(int(data_reader.get_lengths() / config.batch_size)))
            total_xywh_loss = 0
            total_iou_loss = 0
            total_pos_conf_loss = 0
            total_neg_conf_loss = 0
            total_label_loss = 0
            total_loss = 0
            for batch_num in batch_process:
                images = []
                gt_boxes = []
                gt_labels = []
                for i in range(config.batch_size):
                    image, gt_box, gt_label = data_reader.get_items(batch_num * config.batch_size + i)
                    images.append(image)
                    gt_boxes.append(gt_box)  #  [cx, cy, w, h]格式
                    gt_labels.append(gt_label)
                images = np.asarray(images, dtype='float32')
                images = fluid.dygraph.to_variable(images)
                loc_p, conf_p, label_p = model(images)
                xywh_loss, iou_loss, label_loss, pos_conf_loss, neg_conf_loss = yolov3_loss.get_loss((loc_p, conf_p, label_p), gt_boxes, gt_labels, default_anchors)
                loss = xywh_loss + iou_loss + label_loss + pos_conf_loss + neg_conf_loss
                total_xywh_loss += xywh_loss.numpy()[0]
                total_iou_loss += iou_loss.numpy()[0]
                total_pos_conf_loss += pos_conf_loss.numpy()[0]
                total_neg_conf_loss += neg_conf_loss.numpy()[0]
                total_label_loss += label_loss.numpy()[0]
                total_loss += loss.numpy()[0]
                loss.backward()
                optimizer.minimize(loss)
                model.clear_gradients()
                batch_process.set_description_str("epoch:{}".format(epoch))
                batch_process.set_postfix({"x": "{:.3f}".format(xywh_loss.numpy()[0]),
                                           "i":"{:.3f}".format(iou_loss.numpy()[0]),
                                          "l":"{:.3f}".format(label_loss.numpy()[0]),
                                          "p": "{:.3f}".format(pos_conf_loss.numpy()[0]),
                                          "n": "{:.3f}".format(neg_conf_loss.numpy()[0])})
                                        #   "": "{:.3f}".format(loss.numpy()[0])})

            train_logs.write("Epoch:{}, xywh_loss:{:.4f}, iou_loss:{:.4f}, "
                     " pos_conf_loss:{:.4f}, neg_conf_loss:{:.4f}, label_loss:{:.4f}, total_loss:{:.4f} \n".format(epoch, total_xywh_loss / len(batch_process),
                                                                     total_iou_loss / len(batch_process),
                                                                     total_pos_conf_loss / len(batch_process),
                                                                     total_neg_conf_loss / len(batch_process),
                                                                     total_label_loss / len(batch_process),
                                                                     total_loss / len(batch_process)))
            train_logs.flush()
            if epoch % 5 == 0:
                fluid.dygraph.save_dygraph(model.state_dict(), "/home/aistudio/work/code/logs/yolov3" + "_{}".format(epoch))


if __name__ == '__main__':
    yolov3_train()