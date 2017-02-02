package mxnetScala.main

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.optimizer.SGD

class Run {
  // model definition
  val data = Symbol.Variable("data")
  val fc1 = Symbol.FullyConnected(name = "fc1")(Map("data" -> data, "num_hidden" -> 128))
  val act1 = Symbol.Activation(name = "relu1")(Map("data" -> fc1, "act_type" -> "relu"))
  val fc2 = Symbol.FullyConnected(name = "fc2")(Map("data" -> act1, "num_hidden" -> 64))
  val act2 = Symbol.Activation(name = "relu2")(Map("data" -> fc2, "act_type" -> "relu"))
  val fc3 = Symbol.FullyConnected(name = "fc3")(Map("data" -> act2, "num_hidden" -> 10))
  val mlp = Symbol.SoftmaxOutput(name = "sm")(Map("data" -> fc3))

  // load MNIST dataset
  val trainDataIter = IO.MNISTIter(Map(
    "image" -> "data/train-images-idx3-ubyte",
    "label" -> "data/train-labels-idx1-ubyte",
    "data_shape" -> "(1, 28, 28)",
    "label_name" -> "sm_label",
    "batch_size" -> "50",
    "shuffle" -> "1",
    "flat" -> "0",
    "silent" -> "0",
    "seed" -> "10"))

  val valDataIter = IO.MNISTIter(Map(
    "image" -> "data/t10k-images-idx3-ubyte",
    "label" -> "data/t10k-labels-idx1-ubyte",
    "data_shape" -> "(1, 28, 28)",
    "label_name" -> "sm_label",
    "batch_size" -> "50",
    "shuffle" -> "1",
    "flat" -> "0", "silent" -> "0"))

  // setup model and fit the training data
  val model = FeedForward.newBuilder(mlp)
    .setContext(Context.cpu())
    .setNumEpoch(10)
    .setOptimizer(new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f))
    .setTrainData(trainDataIter)
    .setEvalData(valDataIter)
    .build()
  val probArrays = model.predict(valDataIter)
  // in this case, we do not have multiple outputs
  require(probArrays.length == 1)
  val prob = probArrays(0)

  // get real labels
  import scala.collection.mutable.ListBuffer
  valDataIter.reset()
  val labels = ListBuffer.empty[NDArray]
  while (valDataIter.hasNext) {
    val evalData = valDataIter.next()
    labels += evalData.label(0).copy()
  }
  val y = NDArray.concatenate(labels)

  // get predicted labels
  val py = NDArray.argmax_channel(prob)
  require(y.shape == py.shape)

  // calculate accuracy
  var numCorrect = 0
  var numInst = 0
  for ((labelElem, predElem) <- y.toArray zip py.toArray) {
    if (labelElem == predElem) {
      numCorrect += 1
    }
    numInst += 1
  }
  val acc = numCorrect.toFloat / numInst
  println(s"Final accuracy = $acc")
}

object Run{

  def main(args:Array[String]):Unit = {
    println(args(0))
  }
}
