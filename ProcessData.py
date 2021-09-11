# process data
from ProcessData.ProcessData import ProcessData
import os

# Clear
filelist = [ f for f in os.listdir('TrainingData') ]
for f in filelist: os.remove(os.path.join('TrainingData', f))
filelist = [ f for f in os.listdir('ValidationData') ]
for f in filelist: os.remove(os.path.join('ValidationData', f))

# Process
ProcessData(['walk1_subject1.bvh', 'walk3_subject1.bvh', 'run2_subject1.bvh'], 'TrainingData', [[[100, 1400], [2450, 3670]], [[6690, 7330]], [[100, 3000], [3800, -1]]], False)
#ProcessData(['walk1_subject1.bvh', 'run2_subject1.bvh'], 'TrainingData', [[[100, 1400], [2450, 3670]], [[100, 3000], [3800, -1]]], True)

ProcessData(['walk2_subject1.bvh', 'run2_subject1.bvh'], 'ValidationData', [[[100, 730]], [[3000, 3800]]], False)
#ProcessData(['walk2_subject1.bvh', 'run2_subject1.bvh'], 'ValidationData', [[[100, 730]], [[3000, 3800]]], True)

