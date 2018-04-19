# Text recognition

### The purpose of the software is to recognise text from any surface, with uneven lighting.

### Tasks:

1. Identify text segments from a picture
2. Local light invariant (manage to detect text with uneven lighting)
3. Rotation invariant (skewed text, 6 vs 9, d vs p) (handled by classification)
4. Segmentation of symbols
5. Classification of symbols

6. application, read a recipe and recognise important information; how much
did the final sum get to, who did you buy from.


### Approach

- Start (easy) with single sheet of paper with homogenous lighting and clear text.
1. Find rotation of the text, Hough transform
2. Use morphological operations; opening/closing to separate characters
3. Use projection histogram to find border around sentences and then characters
    (remember spaces)
4. Use CNN on the rectangular images, with 0-paddings to get even rectangles
5. Separate words by spaces.

#### Easy
- Numbers only
- Computer text/numbers
- Write an interface/documentation
- binary image

#### Moderate
- Uneven lighting
- English alphabet
- Hand written text
- Use local grey level transformation
- greylevel image

#### Hard
- Detect text anywhere.
- RGB


### Structure
- src
- res
  - datasets
  - images
  - misc
- doc


### Project datasets/libraries
  OpenCV
  TenserFlow
  Numpy



####  Project Overview
##### 1. Find text
INPUT: image RGB
- Text bounder box, rotation on hold

##### 2. Find characters
INPUT: Binary image - matrix with variable size
- Text on white background, straight lines

##### 3. Classification
INPUT: Binary image - matrix with 16x16 pixels (mnist dataset)

- character Classification
