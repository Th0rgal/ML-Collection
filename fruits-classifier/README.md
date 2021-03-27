Everything that follows is based on the work of Isabelle Guyon and Nicolas Thiéry.

# A fruit classifier

## The situation
We need to create a program that is able to sort pictures of apples and bananas (on a light background). We have examples of pictures in ``src/images``.

## How does it work?
First of all, we separated our images into two parts: one part that would be used to create our program and another that would be used to test it.

We then created two features: these are measures that can be extracted from the data (images in this case). These features must be at least a little bit correlated to the type of fruit. Here are the ones that have been chosen:
- redness: which measures the redness of the fruit
- elongation: which measures the lengthening of the fruit
After applying these features to our images, we normalized the results so that they have the same scale. 

For each image in our training set, we have two numbers (which in this case vary between -1.5 and 1.5, but it could be more if there was a banana particularly more elongated than the other fruits for example).

To build our model we finally used the nearest neighbor algorithm. 
I don't think I fully understood how it worked so I'll ask the question next class, but here's my interpretation:
We measured two features for our data, so let's imagine a two-dimensional space. One axis corresponds to redness and another to elongation. For each image of our training set we place a point in this space and give it the label "banana" or "apple". Then when we want to determine the type of another fruit, we just recalculate the features, this gives us a coordinate and we just have to return the label of the closest point to this coordinate in our space.