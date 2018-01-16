# object_detection_api_example
tensorflow object detection api 简单使用的工程化例子
## 环境配置
需要[tensorflow object detection api](https://github.com/tensorflow/models/tree/master/research) 原始api需要先编译protobuf库。我已经编译好了，所以可以直接用。只要object_detection和slim就可以开始使用api了。你可以把这两个文件放在当前目录下也可以加到python的环境变量里去。
## 制作自己的tfrecord数据集
1. 下载labelTool.zip 内有标注说明，使用labelTool工具将原始.jpg图片全部标注为相应的.xml文件。
（标注时所有图片放在同一个文件夹下）标注后将所有.jpg放在data/images/ 所有.xml放在data/annotations/
2. 使用xml_to_csv.py脚本将xml数据转为csv数据（感谢
[datitran](https://github.com/datitran/raccoon_dataset)提供的脚本）然后用spilt_labels.py将数据分为train_labels.csv和test_labels.csv.
3. 调用generate_tfrecord.py，注意要指定--csv_input与--output_path这两个参数。执行下面命令：

```
python generate_tfrecord.py --csv_input=train_labels.csv --output_path=train.record
```
这样就生成了训练及验证用的train.record与test.record。接下来指定标签名称，data/classes_map.pbtxt，填写属于自己的类标签名。
```
item {
  id: 1
  name: 'G0'
}

item {
  id: 2
  name: 'G1'
}
```
## 训练
根据自己的需要，[选择一款用coco数据集预训练的模型](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)（下载模型的时候要加这个前缀http://storage.googleapis.com/否则无法下载成功，可能是google存储位置更改了。）下载 后把前缀是model.ckpt的3个文件放置在training/pre_trained_model/（注意pre_trained_model里只要放3个文件，不要把checkpoint加进去，否则会报错）这里meta文件保存了graph和metadata，ckpt保存了网络的weights，这几个文件表示预训练模型的初始状态。

打开training/ssd_mobilenet_v1_pets.config文件，并做如下修改：

1. num_classes:修改为自己的classes num
```
model {
  ssd {
    num_classes: 10
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }

```
2. 将num_examples设置为自己的测试集数量
```
eval_config: {
  num_examples: 100
}
```
3. （默认情况可忽略这一步骤）将所有PATH_TO_BE_CONFIGURED的地方修改为自己之前设置的路径（共5处）
```
  fine_tune_checkpoint: "training/pre_trained_model/model.ckpt"
  input_path: "data/train.record"
  label_map_path: "data/classes_map.pbtxt"
  input_path: "data/test.record"
  label_map_path: "data/classes_map.pbtxt"
```
准备好上述文件后就可以直接调用train文件进行训练（windows下可能要将/更改为\）。
```
python train.py --logtostderr --pipeline_config_path=training/ssd_mobilenet_v1_pets.config --train_dir=training
```
## tensorboard可视化查看训练状态（可选）
```
tensorboard --logdir=training
```
## Freeze Model模型导出：
查看模型实际的效果前，我们需要用export_inference_graph.py把训练的过程文件导出，产生.pb的模型文件。注：model.ckpt-1000（其中1000为模型保存的训练步数，需要修改）
```
python export_inference_graph.py \
--input_type image_tensor
--pipeline_config_path=training/ssd_mobilenet_v1_pets.config \
--trained_checkpoint_prefix=training/model.ckpt-1000 \
--output_directory=training/export_result
```
## 测试
img_path改为自己的测试图片，直接调用test.py就可以看到效果了。
```
img_path = 'data/images/G5_47.jpg'
```