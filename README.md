# Image-Morphing
Morphs 2 images given the set of correspondences 

## Structure
1. Test images are saved in `images` directory 
2. The corresponding morphed GIFs are in the `results` directory.

## Example
Input images:

![me](/images/Me.jpg)
![Lion](/images/Lion.jpg)

Output morphs:

## Running the code
```
click_correspondences.py
```
Click the corresponding pairs of points on the 2 images (eye to eye, etc.)

```
wrapping.py
```
Calls all the necessary functions to morph the given images as per the correspondences and generates the resulting GIF.
