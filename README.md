# Leaf-architecture

## Structure

'data/networks-BronxA' and 'data/networks-BronxB' contain the original
vectorized leaf networks as value tables. 'data/segments' contains the
leaf segments in pythons pickle format. 'data' itself contains tables to
relate the networks to the corresponding species.

'features' contains tables of the calculated feature values for all leafs.
'features/segments' contains tables of the calculated feature valeus for the
leaf segments.

'scripts/utility.py' contains all the helper functions that we used during the
project, 'scripts/features' the functions used to calculate the features.
'scripts/NET' contains the code of Henrik Ronellenfitsch's NET library that
was used for this project. We made slight modifications where necessary and
removed everything that was not used.

'Documents' contains PDF files that were consulted during the project.

The root folder contains the 'leaf_architecture.ipynb' jupyter notebook which
was used for the data analysis.
