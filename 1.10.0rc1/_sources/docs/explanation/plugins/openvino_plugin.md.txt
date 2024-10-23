
# OpenVINO™ Inference Interpreter

Interpreter samples to parse OpenVINO™ inference outputs.
This section is related to [datumaro/plugins/openvino_plugin](https://github.com/openvinotoolkit/datumaro/tree/develop/datumaro/plugins/openvino_plugin).

## Models supported from interpreter samples
There are detection and image classification examples.

- Detection (SSD-based)
  - Intel Pre-trained Models > Object Detection
    - [face-detection-0200](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_0200_description_face_detection_0200.html)
    - [face-detection-0202](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_0202_description_face_detection_0202.html)
    - [face-detection-0204](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_0204_description_face_detection_0204.html)
    - [person-detection-0200](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_0200_description_person_detection_0200.html)
    - [person-detection-0201](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_0201_description_person_detection_0201.html)
    - [person-detection-0202](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_0202_description_person_detection_0202.html)
    - [person-vehicle-bike-detection-2000](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_vehicle_bike_detection_2000_description_person_vehicle_bike_detection_2000.html)
    - [person-vehicle-bike-detection-2001](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_vehicle_bike_detection_2001_description_person_vehicle_bike_detection_2001.html)
    - [person-vehicle-bike-detection-2002](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_vehicle_bike_detection_2002_description_person_vehicle_bike_detection_2002.html)
    - [vehicle-detection-0200](https://docs.openvinotoolkit.org/latest/omz_models_intel_vehicle_detection_0200_description_vehicle_detection_0200.html)
    - [vehicle-detection-0201](https://docs.openvinotoolkit.org/latest/omz_models_intel_vehicle_detection_0201_description_vehicle_detection_0201.html)
    - [vehicle-detection-0202](https://docs.openvinotoolkit.org/latest/omz_models_intel_vehicle_detection_0202_description_vehicle_detection_0202.html)

  - Public Pre-Trained Models(OMZ) > Object Detection
    - [ssd_mobilenet_v1_coco](https://docs.openvinotoolkit.org/latest/omz_models_public_ssd_mobilenet_v1_coco_ssd_mobilenet_v1_coco.html)
    - [ssd_mobilenet_v2_coco](https://docs.openvinotoolkit.org/latest/omz_models_public_ssd_mobilenet_v2_coco_ssd_mobilenet_v2_coco.html)

- Image Classification
  - Public Pre-Trained Models(OMZ) > Classification
    - [mobilenet-v2-pytorch](https://docs.openvinotoolkit.org/latest/omz_models_public_mobilenet_v2_pytorch_mobilenet_v2_pytorch.html)

You can find more OpenVINO™ Trained Models
[here](https://docs.openvinotoolkit.org/latest/omz_models_intel_index.html)
To run the inference with OpenVINO™, the model format should be Intermediate
Representation(IR).
For the Caffe/TensorFlow/MXNet/Kaldi/ONNX models, please see the
[Model Conversion Instruction](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html)

You need to implement your own interpreter samples to support the other
OpenVINO™ Trained Models.

## Model download

Prerequisites:
- OpenVINO™ (To install OpenVINO™, please see the
  [OpenVINO™ Installation Instruction](https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html))
- OpenVINO™ models (To download OpenVINO™ models, please see the [Model Downloader Instruction](https://docs.openvinotoolkit.org/latest/omz_tools_downloader_README.html))
- PASCAL VOC 2012 dataset (To download VOC 2012 dataset, please go [VOC2012 download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit))

Open Model Zoo models can be downloaded with the Model Downloader tool
from OpenVINO™ distribution:

```bash
cd <openvino_dir>/deployment_tools/open_model_zoo/tools/downloader
./downloader.py --name <model_name>
```

Example: download the "face-detection-0200" model
```bash
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
./downloader.py --name face-detection-0200
```

## Model inference

Prerequisites:
- OpenVINO™ (To install OpenVINO™, please see the
  [OpenVINO™ Installation Instruction](https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html))
- Datumaro (To install Datumaro, please see the [user manual](../../get-started/quick-start-guide/installation.rst))
- OpenVINO™ models (To download OpenVINO™ models, please see the [Model Downloader Instruction](https://docs.openvinotoolkit.org/latest/omz_tools_downloader_README.html))
- PASCAL VOC 2012 dataset (To download VOC 2012 dataset, please go [VOC2012 download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit))

## Examples

To run the inference with OpenVINO™ models and the interpreter samples,
please follow the instructions below.

```bash
source <openvino_dir>/bin/setupvars.sh
datum project create -o <proj_dir>
datum model add -l <launcher> -p <proj_dir> --copy -- \
  -d <path/to/xml> -w <path/to/bin> -i <path/to/interpreter/script>
datum project import -p <proj_dir> -f <format> <path_to_dataset>
datum model run -p <proj_dir> -m model-0
```

### Detection: ssd_mobilenet_v2_coco

```bash
source /opt/intel/openvino/bin/setupvars.sh
cd datumaro/plugins/openvino_plugin
datum project create -o proj
datum model add -l openvino -p proj --copy -- \
    --output-layers=do_ExpandDims_conf/sigmoid \
    -d model/ssd_mobilenet_v2_coco.xml \
    -w model/ssd_mobilenet_v2_coco.bin \
    -i samples/ssd_mobilenet_coco_detection_interp.py
datum project import -p proj -f voc VOCdevkit/
datum model run -p proj -m model-0
```

### Classification: mobilenet-v2-pytorch

```bash
source /opt/intel/openvino/bin/setupvars.sh
cd datumaro/plugins/openvino_plugin
datum project create -o proj
datum model add -l openvino -p proj --copy -- \
    -d model/mobilenet-v2-pytorch.xml \
    -w model/mobilenet-v2-pytorch.bin \
    -i samples/mobilenet_v2_pytorch_interp.py
datum project import -p proj -f voc VOCdevkit/
datum model run -p proj -m model-0
```
