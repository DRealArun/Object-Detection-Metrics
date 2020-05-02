###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
# Last modification: May 2nd 2020 (Arun Rajendra Prabhu : Hochschule Bonn Rhein Sieg)     #
###########################################################################################

import argparse
import glob
import os
import shutil
# from argparse import RawTextHelpFormatter
import sys

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat


# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(
            'argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' % argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append('argument %s: required argument' % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append(
                '%s. It must be in the format \'width,height\' (e.g. \'600,400\')' % errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
        os.makedirs(arg)
        #errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg


def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            poly = []
            if isGT:
                idClass = (splitLine[0])  # class
                if len(splitLine[1:]) > 4:
                    coords = [float(v) for v in splitLine[1:]]
                    for l in range(0,len(coords),2):
                        poly.append((float(coords[l]), float(coords[l+1])))
                    x = float(min(coords[0::2]))
                    y = float(min(coords[1::2]))
                    w = float(max(coords[0::2]))
                    h = float(max(coords[1::2]))
                    coordType = CoordinatesType.Absolute
                    bbFormat = BBFormat.XYX2Y2
                else:
                    x = float(splitLine[1])
                    y = float(splitLine[2])
                    w = float(splitLine[3])
                    h = float(splitLine[4])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.GroundTruth,
                    format=bbFormat,
                    poly=poly)
            else:
                idClass = (splitLine[0])  # class
                confidence = float(splitLine[1])
                if len(splitLine[2:]) > 4:
                    coords = [float(v) for v in splitLine[2:]]
                    for l in range(0,len(coords),2):
                        poly.append((float(coords[l]), float(coords[l+1])))
                    x = float(max(min(coords[0::2]), 0))
                    y = float(max(min(coords[1::2]), 0))
                    w = float(min(max(coords[0::2]), 1024-1))
                    h = float(min(max(coords[1::2]), 512-1))
                    width = w - x
                    height = h - y
                    cx = x + 0.5*width
                    cy = y + 0.5*height
                    x = round(cx - (width/2), 2)
                    y = round(cy - (height/2), 2)
                    w = round(cx + (width/2), 2)
                    h = round(cy + (height/2), 2)   
                    coordType = CoordinatesType.Absolute
                    bbFormat = BBFormat.XYX2Y2
                else:
                    idClass = (splitLine[0])  # class
                    confidence = float(splitLine[1])
                    x = float(splitLine[2])
                    y = float(splitLine[3])
                    w = float(splitLine[4])
                    h = float(splitLine[5])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.Detected,
                    confidence,
                    format=bbFormat,
                    poly=poly)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses


# Get current path to set default folders
currentPath = os.path.dirname(os.path.abspath(__file__))

VERSION = '0.1 (beta)'

parser = argparse.ArgumentParser(
    prog='Object Detection Metrics - Pascal VOC',
    description='This project applies the most popular metrics used to evaluate object detection '
    'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
    'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
    epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
# formatter_class=RawTextHelpFormatter)
parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
# Positional arguments
# Mandatory
parser.add_argument(
    '-root',
    '--root_folder',
    dest='rootFolder',
    default=os.path.join(currentPath),
    metavar='',
    help='folder containing multiple folders of ground truth and detection bounding boxes')
parser.add_argument(
    '-gtfiles',
    '--gt_files_folder',
    dest='gtFolder',
    default=os.path.join(currentPath),
    metavar='',
    help='folder containing gt files which are used when calculation mAP of box vs mask')
parser.add_argument(
    '-gtcoords',
    dest='gtCoordinates',
    default='abs',
    metavar='',
    help='reference of the ground truth bounding box coordinates: absolute '
    'values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument(
    '-detcoords',
    default='abs',
    dest='detCoordinates',
    metavar='',
    help='reference of the ground truth bounding box coordinates: '
    'absolute values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument(
    '-imgsize',
    dest='imgSize',
    metavar='',
    help='image size. Required if -gtcoords or -detcoords are \'rel\'')
parser.add_argument(
    '-np',
    '--noplot',
    dest='showPlot',
    action='store_false',
    help='no plot is shown during execution')
parser.add_argument(
    '-mask',
    '--usemask',
    dest='useMask',
    action='store_true',
    help='convert to binary masks before iou calculation?')
parser.add_argument(
    '-crop',
    '--cropMask',
    dest='cropMask',
    action='store_true',
    help='crop the masks to save time during iou calculation')
args = parser.parse_args()

# Arguments validation
errors = []
# Groundtruth folder
if ValidateMandatoryArgs(args.rootFolder, '-root/--root_folder', errors):
    rootFolder = ValidatePaths(args.rootFolder, '-root/--root_folder', errors)
else:
    # errors.pop()
    rootFolder = os.path.join(currentPath, 'root')
    if os.path.isdir(rootFolder) is False:
        errors.append('folder %s not found' % rootFolder)
# Coordinates types
gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
detCoordType = ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
imgSize = (0, 0)
if gtCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
if detCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)
if args.useMask:
    imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-useMask', errors)

# If error, show error messages
if len(errors) is not 0:
    print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                [-detformat] [-save]""")
    print('Object Detection Metrics: error(s): ')
    [print(e) for e in errors]
    sys.exit()

# Show plot during execution
showPlot = args.showPlot

eval_directories = []
eval_directories.extend([x[0].split('detections')[0] for x in os.walk(rootFolder) if 'detections' in x[0]])
reject_eval_directories = [x[0].split('mask')[0] for x in os.walk(rootFolder) if 'mask' in x[0]] # Check if any directory is already processed.
for d in reject_eval_directories:
    if d in eval_directories:
        eval_directories.remove(d)
print("Number of evaluation directories:", len(eval_directories))
print('gtCoordType = %s' % gtCoordType)
print('detCoordType = %s' % detCoordType)
print('showPlot %s' % showPlot)
print('cropMask %s' % args.cropMask)
print('useMask %s' % args.useMask)

for useGtBBoxMask, useDtBBoxMask in [[True, True], [False, True], [False, False]]:
    for eval_dir in eval_directories:
        if '.ipynb_checkpoints' in eval_dir:
            continue
        gtFolder = os.path.join(eval_dir, 'groundtruths')
        detFolder = os.path.join(eval_dir, 'detections')
        # Validate formats
        if 'xywh' in eval_dir:
            gtFormat = ValidateFormats('xywh', '-gtformat', errors)
            detFormat = ValidateFormats('xywh', '-detformat', errors)
        else: # if xwrb in eval_dir or coords in eval_dir
            gtFormat = ValidateFormats('xyrb', '-gtformat', errors)
            detFormat = ValidateFormats('xyrb', '-detformat', errors)
        print('Processing %s' % eval_dir)
        print('detFolder = %s' % detFolder)
        print('gtFormat = %s' % gtFormat)
        print('detFormat = %s' % detFormat)
        for iouThreshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        # for iouThreshold in [0.3]:
            print('iouThreshold= %f' % iouThreshold)
            if args.useMask:
                if 'coords' in eval_dir:
                    if useGtBBoxMask and useDtBBoxMask:
                        savePath = os.path.join(eval_dir, 'mask', 'both4', 'results_'+str(iouThreshold))
                    elif useGtBBoxMask:
                        savePath = os.path.join(eval_dir, 'mask', 'onlyGtmask', 'results_'+str(iouThreshold))
                    elif useDtBBoxMask:
                        savePath = os.path.join(eval_dir, 'mask', 'onlyDtmask', 'results_'+str(iouThreshold))
                    else:
                        savePath = os.path.join(eval_dir, 'mask', 'both8', 'results_'+str(iouThreshold))
                else:
                    if useGtBBoxMask and useDtBBoxMask:
                        savePath = os.path.join(eval_dir, 'mask', 'both4', 'results_'+str(iouThreshold))
                    elif useDtBBoxMask:
                        # box2mask map calculation
                        assert os.path.exists(args.gtFolder)
                        gtFolder = args.gtFolder
                        savePath = os.path.join(eval_dir, 'mask', 'box2mask', 'results_'+str(iouThreshold))
                    else:
                        print("Skipping because gtbboxmask:", useGtBBoxMask, "detbboxmask:", useDtBBoxMask, "for iouThreshold:", iouThreshold)
                        continue
            else:
                savePath = os.path.join(eval_dir, 'no_mask', 'results_'+str(iouThreshold))

            print('gtFolder = %s' % gtFolder)
            print('useGtBBoxMask %s' % useGtBBoxMask)
            print('useDtBBoxMask %s' % useDtBBoxMask)
            print('savePath = %s' % savePath)        
            # Create directory to save results
            if os.path.exists(savePath):
                shutil.rmtree(savePath, ignore_errors=True)  # Clear folder
            os.makedirs(savePath)

            # Get groundtruth boxes
            allBoundingBoxes, allClasses = getBoundingBoxes(gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize)
            # Get detected boxes
            allBoundingBoxes, allClasses = getBoundingBoxes(
                detFolder, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
            allClasses.sort()

            evaluator = Evaluator()
            acc_AP = 0
            validClasses = 0

            # Plot Precision x Recall curve
            detections = evaluator.PlotPrecisionRecallCurve(
                allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
                IOUThreshold=iouThreshold,  # IOU threshold
                method=MethodAveragePrecision.EveryPointInterpolation,
                showAP=True,  # Show Average Precision in the title of the plot
                showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
                savePath=savePath,
                showGraphic=showPlot,
                useMask=args.useMask,
                useBBoxMask=[useGtBBoxMask, useDtBBoxMask],
                cropMask=args.cropMask)

            f = open(os.path.join(savePath, 'results.txt'), 'w')
            f.write('Object Detection Metrics\n')
            f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
            f.write('Average Precision (AP), Precision and Recall per class:')

            # each detection is a class
            for metricsPerClass in detections:

                # Get metric values per each class
                cl = metricsPerClass['class']
                ap = metricsPerClass['AP']
                precision = metricsPerClass['precision']
                recall = metricsPerClass['recall']
                totalPositives = metricsPerClass['total positives']
                total_TP = metricsPerClass['total TP']
                total_FP = metricsPerClass['total FP']

                if totalPositives > 0:
                    validClasses = validClasses + 1
                    acc_AP = acc_AP + ap
                    prec = ['%.2f' % p for p in precision]
                    rec = ['%.2f' % r for r in recall]
                    ap_str = "{0:.2f}%".format(ap * 100)
                    # ap_str = "{0:.4f}%".format(ap * 100)
                    print('AP: %s (%s)' % (ap_str, cl))
                    f.write('\n\nClass: %s' % cl)
                    f.write('\nAP: %s' % ap_str)
                    f.write('\nPrecision: %s' % prec)
                    f.write('\nRecall: %s' % rec)

            mAP = acc_AP / validClasses
            mAP_str = "{0:.2f}%".format(mAP * 100)
            print('mAP: %s' % mAP_str)
            f.write('\n\n\nmAP: %s' % mAP_str)