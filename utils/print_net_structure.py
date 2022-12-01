import  net_generator
import mxnet as mx 

net = net_generator.get_alexnet(2)
mx.viz.print_summary(net,shape={"data":(1,3,128,128)})

internals = net.get_internals()
print(internals['flatten0_output'])
