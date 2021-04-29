# Spatial Statistical Analysis of Cell Images

Current method to describe spatial distribution within cell images: Use Ripley's K function to compare the actual distribution of marked points in the cell with Complete Spatial Randomness (CSR).



1. Load .tif file containing 3D image of cell.
2. Select xy-slice that shows the biggest part of the cell.
3. Set all pixels outside the cell area to zero intensity.
4. Create a "CSR" image in which every pixel in the cell is assigned the average intensity (of pixels within the cell).
5. Compute and compare Ripley's K function for the real image and the CSR image.



(More explanations + equations will follow)