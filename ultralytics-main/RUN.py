from ultralytics import YOLO
if __name__ == '__main__':


    model = YOLO("yolov8l.yaml")  # 从头开始构建新模型

    result= model.train(data='G:/yolov8/ultralytics-main/datesets/voc2/mydata.yaml', pretrained ='yolov8l.pt')# 训练模型

#val--------------------------------------------------

   # model = YOLO("G:/yolov8/ultralytics-main/runs/detect/v8n+nam0.655/weights/best.pt")  # 权重地址
    #results = model.val(data="G:/yolov8/ultralytics-main/datesets/voc2/mydata.yaml")  # 参数和训练用到的一样



# yolo task=detect mode=train model=yolov8l.pt data=G:/yolov8/ultralytics-main/datesets/voc2/mydata.yaml pretrained =yolov8l.pt batch=4 epochs=150
