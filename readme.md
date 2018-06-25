# Hierachical softmax layer

Hierachical softmax layer can be implemented with a combination of Slice, Tile, and Eltwise layers. This hierarchisoftmax layer combines them together.

The hierarchisoftmax layer has one parameters which provides the idex of its parent caetgory (idex in higher level softmax output).

# How to use this layer 

To use the hierarchical softmax layer, add following into the proto file
~~~~
optional HierarchiSoftMaxParameter hierarchi_softmax_param = 501;
~~~~
~~~~
message HierarchiSoftMaxParameter {
  optional uint32 cat_id = 1 [default =1]; // parent cat id, start from 0
}
~~~~
