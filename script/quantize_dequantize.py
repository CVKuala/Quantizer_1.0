from collections import namedtuple
QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def calcScaleZeroPoint(min_val, max_val,num_bits=8):
  # Calc Scale and zero point of next 
  qmin = 0.
  qmax = 2.**num_bits - 1.

  scale = (max_val - min_val) / (qmax - qmin)

  initial_zero_point = qmin - min_val / scale
  if(scale==0):
     scale=1
  
  zero_point = 0
  if initial_zero_point < qmin:
      zero_point = qmin
  elif initial_zero_point > qmax:
      zero_point = qmax
  else:
      zero_point = initial_zero_point

  zero_point = int(zero_point)

  return scale, zero_point




def dequantize_tensor(q_x):
  return q_x.scale * (q_x.tensor - q_x.zero_point)				



def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
      min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x = np.clip(q_x,qmin,qmax)
    q_x = q_x.round()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)




def quantizeLayer(x, layer, stat, scale_x, zp_x, net):

  index=list(net._layer_names).index(layer)  		
  # for both conv and linear layers

  # cache old values
  W = net.params[layer][0].data 		
  B = net.params[layer][1].data			

  # quantise weights, activations are already quantised
  w = quantize_tensor(net.params[layer][0].data) 		
  b = quantize_tensor(net.params[layer][1].data)		

  net.params[layer][0].data[...] = w.tensor			
  net.params[layer][1].data[...] = b.tensor			

  # This is Quantisation Artihmetic
  scale_w = w.scale
  zp_w = w.zero_point
  scale_b = b.scale
  zp_b = b.zero_point
  
  scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

  # Preparing input by shifting
  X = x - zp_x											
  net.params[layer][0].data[...] = scale_x * scale_w*(net.params[layer][0].data - zp_w)		
  net.params[layer][1].data[...] = scale_b*(net.params[layer][1].data + zp_b)			

  if (index-1==0):
	net.blobs['data'].data[...]=X
  else:
	net.blobs[list(net._layer_names)[index-1]].data[...]=X				
  

  # All int computation
  x = (net.forward(start=layer,end=layer)[layer]/ scale_next) + zero_point_next 		
  
  # Perform relu too
  ##x = F.relu(x)						

  # Reset weights for next forward pass
  net.params[layer][0].data[...] = W  					
  net.params[layer][1].data[...] = B					
  
  return x, scale_next, zero_point_next




def updateStats(x, stats, key):
  max_val = np.max(x)			
  min_val = np.min(x)			
  
  
  if key not in stats:
    stats[key] = {"max": max_val, "min": min_val, "total": 1}		
  else:
    stats[key]['max'] += max_val			
    stats[key]['min'] += min_val			
    stats[key]['total'] += 1
  
  return stats





def gatherActivationStats(model, x, stats):

  model.blobs['data'].data[...]=x						

  stats = updateStats(x.reshape(x.shape[0], -1), stats, 'conv1')			
  
  x=model.forward(start='conv1',end='conv1')['conv1']				
  x = model.forward(start='pool1',end='pool1')['pool1']					
  
  stats = updateStats(x.reshape(x.shape[0], -1), stats, 'conv2')			
  
  x=model.forward(start='conv2',end='conv2')['conv2']				
  x = model.forward(start='pool2',end='pool2')['pool2']					
  x = x.reshape(-1, 4*4*50)							
  
  stats = updateStats(x, stats, 'ip1')					

  x = model.forward(start='ip1',end='ip1')['ip1']			
  x = model.forward(start='relu1',end='relu1')['ip1']				
  
  stats = updateStats(x, stats, 'ip2')					

  x = model.forward(start='ip2',end='ip2')['ip2']			

  return stats





def gatherStats(model, test_loader):					
    device = 'cuda'
    
    ##model.eval()
    test_loss = 0
    correct = 0
    stats = {}
    									
    for data, target in test_loader:					
       ##data, target = data.to(device), target.to(device)		
       stats = gatherActivationStats(model, data, stats)
    
    final_stats = {}
    for key, value in stats.items():
      final_stats[key] = { "max" : value["max"] / value["total"], "min" : value["min"] / value["total"] }
    return final_stats






def quantForward(model, x, stats):
  
  # Quantise before inputting into incoming layers
  x = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])

  x, scale_next, zero_point_next = quantizeLayer(x.tensor, 'conv1', stats['conv2'], x.scale, x.zero_point,model)	 

  x = net.forward(start='pool1',end='pool1')['pool1']					
  
  x, scale_next, zero_point_next = quantizeLayer(x, 'conv2', stats['ip1'], scale_next, zero_point_next,model)		

  x = net.forward(start='pool2',end='pool2')['pool2']					

  x = x.reshape(64,50,4,4)								

  x, scale_next, zero_point_next = quantizeLayer(x, 'ip1', stats['ip2'], scale_next, zero_point_next,model)

  x = net.forward(start='relu1',end='relu1')['ip1']				
  
  # Back to dequant for final layer
  x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
   
  x = net.forward(start='ip2',end='ip2')['ip2']						

  return net.forward(start='loss',end='loss')['loss']					




def testQuant(model, test_loader, quant=False, stats=None):
    device = 'cuda'
    
    ##model.eval()
    test_loss = 0
    correct = 0
    								
    for data, target in test_loader:
        ##data, target = data.to(device), target.to(device)		
        if quant:
           output = quantForward(model, data, stats)
        else:
           output = model(data)
        ##test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = np.argmax(output['prob'],dim=1) # get the index of the max log-probability		
        ##correct += pred.eq(target.view_as(pred)).sum().item()						
	for i in range(len(pred)):
	    if(pred[i]==target[i]):
		correct+=1

    ##test_loss /= len(test_loader.dataset)

    print('Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




ACTUAL CODE---

##import copy
##q_model = copy.deepcopy(model)

import caffe
caffe.set_mode_cpu()

net=caffe.Net("example.prototxt",caffe.TEST)					


kwargs = {'num_workers': 1, 'pin_memory': True}
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, **kwargs)



testQuant(net, test_loader, quant=False)  ## for non quantized version   	


stats = gatherStats(net, test_loader)   ## for stats					
print(stats)


testQuant(net, test_loader, quant=True, stats=stats)  ## for quantized version		####
