#!/usr/bin/python3

import sys
import argparse

# print("warning:  importing Jetson.Inference is deprecated.  please 'import jetson_inference' instead.")
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("path", type=str, default="", nargs='?', help="The path to onnx model")
parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video sources and outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

net = detectNet(argv=[f"--model={args.path}/ssd-mobilenet.onnx", f"--labels={args.path}/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"], threshold=0.5)

while True:
    img = input.Capture()

    if img is None: # capture timeout
        continue

    detections = net.Detect(img)
    print(detections)
    
    output.Render(img)
    output.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
