# train-tf-object-detection

## 1. Label images 

  - If you are using a pre-labeled dataset, you may have to jump forward to one of the steps between 2 to 6 depending on the format of the labels.
  
  - If you are using [label-V](https://github.com/innovationgarage/label-V), you may jump to step 3.

  - Follow these steps to make [ImageNet-Utils](https://github.com/tzutalin/ImageNet_Utils)
  
        git clone --recursive https://github.com/tzutalin/ImageNet_Utils.git  
        mkvirtualenv Imagenet_utils
        workon Imagenet_utils
        sudo apt-get install pyqt4-dev-tools
        pip install Pillow
        cd ImageNet_Utils/labelImgGUI
        make all
    
  - Now run the labeling tool
  
        ./labelImg.py
    
  - Labels will be by default saved in the same directory as images. This could be set differently when saving each label but I forgot to do that. So I had to clean up afterwards.
  
        mkdir labels
        mv images/*.xml labels/
  
## 2. Change the format of labels
  
  - Now run the script below from the root directory to make a list of all bboxes. The list will be written to __labels/all_labels.csv__
  
        python xml2csv.py
         
## 3. Check [some of the ] bboxes

  - Install the required libraries
  
        pip install opencv-contrib-python
        pip install matplotlib
      
  - Run the following script to check the labeling you did. The output will be saved in __training_data.png__ 
  
        python draw_training_data.py

## 4. Split training/validation sets

  - Run the following commands to make __data/train_label.csv__ and __data/test_label.csv__ using the test fraction as the argument to the split function.
  
        mkdir data
        python split_train_valid.py 0.05
     
## 5. Generate TFRecords for training/validation sets

  - At this stage we need to use the tensorflow object detection, so I copy the entire directory to my project directory and then the images, labels and data to __tensorflow/models__
  
        mkdir tensorflow
        cd tensorflow
        mkdir models
        cd models
        rsync -r --progress ~/tf-models/research/object_detection/ object_detection/
        mv ../../generate_tfrecord.py .
        rsync -r --progress ../../images/ Images/
        rsync -r --progress ../../data/ data/
       
  - I have a separate virtualenv for tf object detection, so I switch to work in that environment from now on
  
        workon tf-objectdetection
        
  - Make sure yo introduce the map for your labels, before generating TFRecords from the csv file. The format of this map simply needs to be as below for each class:

        item {
          id: 1
          name: 'class label'
        }
        
  - As of now, you also need to modify function __class_text_to_int__ in __generate_tfrecord.py__ to match the class name as well (__FIXME!__)        

  - Then run the following for  __train_label.csv__ and __test_label.csv__ to convert them to TFrecords (the format TF expects to read image labels)
  
        python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
        python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
        
## 6. Configure the network and choose a checkpoint to use

  - Find the config file you find best fitting to your problem or use the one included in thie repository. I am running my training locally and therefore I need to set the path to traininig/validation records, as well as the model checkpoint (let's say __ssd_mobilenet_v1_coco__). if you use my config file search for the term __PATH_TO_BE_CONFIGURED__ and set the paths. Also make sure to set the correct value to __num_classes__!
  
## 7. Train

  - Run the following line and make sure to pass the correct arguments

        python object_detection/train.py \
            --logtostderr \
            --pipeline_config_path=./ssd_mobilenet_v1_coco.config \
            --train_dir=./model/train/

      
## 8. Validate

  - Run this in a separate window (make sure to change to the correct virtualenv first)
  
        python object_detection/eval.py \
            --logtostderr \
            --pipeline_config_path=./ssd_mobilenet_v1_coco.config \
            --checkpoint_dir=./model/train/ \
            --eval_dir=/model/valid/
        
## 9. Check the progress

  - Now let's run a tensorboard instance to check the progress for both training and validation sets.
  
        tensorboard --logdir=./model/
        
## 10. Export the trained model

  - Once training is finished, you may want to export the latest checkpoint of your model (or any checkpoint for that matter) to a model and use it for a later training as the base.
  
        python object_detection/export_inference_graph.py \
            --input_type image_tensor \
            --pipeline_config_path=./ssd_mobilenet_v1_coco.config \
            --trained_checkpoint_prefix ./model/train/model.ckpt-[EPOCH NO] \
            --output_directory .checkpoitns/saved_model_[EPOCH NO].pb
      
