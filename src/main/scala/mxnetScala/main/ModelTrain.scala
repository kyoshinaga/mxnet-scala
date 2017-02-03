package mxnetScala.main

import ml.dmlc.mxnet.Callback.Speedometer
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.optimizer._
import org.slf4j.LoggerFactory

object ModelTrain {

  private val logger = LoggerFactory.getLogger(classOf[ModelTrain])

  def fit(dataDir: String, batchSize: Int, numExamples: Int, devs: Array[Context],
          network: Symbol, dataLoader: (String, Int, KVStore) => (DataIter, DataIter),
          kvStore: String, numEpochs: Int, modelPrefix: String = null, loadEpoch: Int = -1,
          lr: Float = 0.1f, lrFactor: Float = 1f, lrFactorEpoch: Float = 1f,
          clipGradient: Float = 0f, opt: String = "sgd"): Unit = {
    //kvstore
    //TODO: if local model and no gpu is used set kv = null
    val kv = KVStore.create(kvStore)

    // load model
    val modelPrefixWithRank =
      if (modelPrefix == null) null
      else modelPrefix + s"-${kv.rank}"

    val (argParams,auxParams, beginEpoch) =
      if(loadEpoch >= 0) {
        require(modelPrefixWithRank != null)
        val tmp = FeedForward.load(modelPrefix, loadEpoch)
        (tmp.getArgParams, tmp.getAuxParams, loadEpoch)
      }else{
        (null, null, 0)
      }
    // save model
    val checkpoint: EpochEndCallback =
      if(modelPrefix == null) null
      else new EpochEndCallback {
        override def invoke(epoch: Int, symbol: Symbol,
                            argParams: Map[String, NDArray],
                            auxStates: Map[String, NDArray]): Unit = {
          Model.saveCheckpoint(modelPrefix, epoch + 1, symbol, argParams, auxParams)
        }
      }

    // data
    val (train, validation) = dataLoader(dataDir, batchSize, kv)

    // train
    val epochSize =
      if(kvStore == "dist_sync") numExamples / batchSize / kv.numWorkers
      else numExamples / batchSize

    val lrScheduler =
      if(lrFactor < 1f){
        new FactorScheduler(step = Math.max((epochSize * lrFactorEpoch).toInt, 1),
          factor = lrFactor)
      }else{
        null
      }

    val optimizer: Optimizer =
      opt.toLowerCase match {
        case "adadelta" =>
          new AdaDelta(rho = lr, clipGradient= clipGradient, wd = 0.00001f)
        case "adagrad" =>
          new AdaGrad(learningRate = lr, wd = 0.00001f)
        case "adam" =>
          new Adam(wd = 0.00001f)
        case "rmsprop" =>
          new RMSProp(wd = 0.00001f)
        case "sgd" =>
          new SGD(learningRate = lr, lrScheduler = lrScheduler,
            clipGradient = clipGradient, momentum=0.9f, wd = 0.00001f)
        case other =>
          throw new IllegalArgumentException(s"Unsupported optimizer $other")
      }

    val model = new FeedForward(ctx = devs,
      symbol = network,
      numEpoch = numEpochs,
      optimizer = optimizer,
      initializer = new Xavier(factorType = "in", magnitude = 2.34f),
      argParams = argParams,
      auxParams = auxParams,
      beginEpoch = beginEpoch,
      epochSize = epochSize)

    model.fit(trainData = train, evalData = validation, evalMetric = new Accuracy(),
      kvStore = kv,
      batchEndCallback = new Speedometer(batchSize, 50),
      epochEndCallback = checkpoint)

    kv.dispose()
  }
  // scalastyle: on parameterNum
}

class ModelTrain