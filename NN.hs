module NN where

import Shuffle
import LinearAlg
import Types
import System.Random
import Data.Tuple --swap

-- PRELIMINARIES --

weights :: NetworkWeights -> [LayerWeights]
weights = map fst

epsilon0 :: TrainingRate
epsilon0 = 0.01

meanSquaredError':: LossFunction'
meanSquaredError' target predicted = vecVecDiff target predicted

-- could generalize with a buildNetwork Function
sigmaNetwork :: NetworkWeights -> Network
sigmaNetwork n = (n,sigmoid)

-- >>> map sigmoid ((-5) : 0 : [5])
-- [6.692851e-3,0.5,0.9933072]
sigmoid :: ActivationFcn
sigmoid x = 1 / (1 + exp (- x))

-- notice that we can just take advantage of the fact that
-- sigma ' = sigma (1 - sigma)
-- this way enables significant speedup from
-- applying the actual sigmoid again during backprop
sigmoid' :: ActivationFcn'
sigmoid' output = vecVecMul output (vecVecDiff n1s output)
  where
    n1s = replicate (length output) 1

-- activation, a^l, and weighted input, z^l, at layer l
-- we compute these lists so that they can be efficently recalled during backprop
singleLayer :: Input -> Layer -> ActivationFcn -> (WeightedInput,Activation)
singleLayer ins (weights,bias) activate =
  let weighted = (affineTransform weights ins bias)
      activation = map activate weighted -- refactor renaming 'map activate'
  in (weighted,activation)

-- Section 2.6 , The Backpropagation Agorithm --

-- notice that we accumulate both the weighted inputs and activations for all
-- layers, with the intention being that these will be stored for usage during
-- backprop
feedForward :: Input -> Network -> [(WeightedInput,Activation)]
feedForward ins ([],activate) = [(ins,ins)] -- ERROR I think this should be empty
feedForward ins ((l:ls),activate) =
  let out@(weightedOut,activatedOut) = singleLayer ins l activate
  in out : (feedForward activatedOut (ls,activate))

-- z weighted input, a acitvation
--compute
computeOutputError ::
  Output         ->
  WeightedInput  ->
  Activation     ->
  LossFunction'  ->
  ActivationFcn' ->
  LayerError
computeOutputError out z a lossF actvF = vecVecMul (lossF out a) (actvF z)

-- TODO transpose all the matrices for backprop
-- Depth might be unncessary, should be able to recurse on number of
-- errors and weighted inputs shou
-- so this network ends at the "beginning" of the network
-- we've luckily avoided the last layer tho
backPropError ::
  LayerError      -> -- initial error
  -- Depth           ->    -- could decrease from depth of the network, implicit
  ActivationFcn'  ->
  [LayerWeights] ->
  [WeightedInput] ->
  [LayerError]
backPropError delta_L sigma' ws [] = [] -- outputError
backPropError delta_L sigma' (ws:wss) (z:zs) =
  let delta_l :: LayerError -> LayerError --transpose needed over ws
      delta_l delta_lPlus1 = vecVecMul (matVectProd ws delta_lPlus1) (sigma' z)
      delta_l0 = delta_l delta_L
  in delta_l0 : (backPropError delta_l0 sigma' wss zs)

generateGradients ::
  [LayerError] ->
  [Activation] -> (NetworkWeightGradients,NetworkBiasGradients)
generateGradients delta_s a_s = (zipWith weightupdates delta_s a_s,delta_s )
  where
    reverse_a_s = reverse a_s
    weightupdates delta a = matMatProd (vec2Mat delta) (vec2Mat a)

computeGradient ::
  (Input,Output) ->
  Network ->
  LossFunction'  ->
  ActivationFcn' ->
  (NetworkWeightGradients,NetworkBiasGradients)
computeGradient (x,target) n@(weightsAndBiases,_) loss' sigma' =
  let wghtdInputsAndActvns = feedForward x n
      weightedInputs       = getWeightedInputs wghtdInputsAndActvns
      activations          = getActivations wghtdInputsAndActvns
      outputError          = computeOutputError
                               target
                               (head weightedInputs)
                               (head activations)
                               loss'
                               sigma'
      transposeWeights     = map transpose (weights weightsAndBiases)
      networkErrors        = backPropError
                               outputError
                               sigma'
                               transposeWeights
                               weightedInputs
  in generateGradients networkErrors activations

{-
Elaborating Stochastic Gradient Desent, at least the specific implementation
we've impelemented here. Stochastic gradient descent can be seen as an algorithm
for deriving parameters giving an optimum to a loss function seeking to minimize
the desired error with regards to the input output behavior of some set of data.
While one could seek analytic solutions to these optimization problems,
numerical methods reign supreme, and universally so when the data comes from
real world sources equipped with noise and bias. "Learning" can then be meant as
identifying patterns in the data via this optimization technique that typically
have been associated with intelligent agents, i.e. humans. And "neural networks"
are one class of "computational learners" being used to identify such patterns,
the analogy being that learning is a cognitive capability, and that the neurons
of the brain, arranged in networks, should be reflected somehow in the
computational learning problem.

The networks feeding the loss functions are composed of layers of weights
matrices, bias vectors, and nonlinear activation functions. The optimal
parameters of the model, which allow for the minimal error between the trained
model output after being fed some input vector relative to its correctly labeled
output.

Given a network with sone weights W_{i,j} \in R^n\xR^m The data moves forward
through the network via affine transformations, followed by component-wise
applications of the nonlinear function, its intermediare states being being
stored or memoized for later. Once the final transformation has been reached,
the output training value is compared with the target vector, whereby their
error of their difference, or the function defined as such, can be minizimed by
taking its gradient, calculating weight and bias derivatives, and updating the
weights. The gradient calcuation, known as backpropagation, allows one to apply
the chain rule to the chains of composed forward units, and thereby find how
small perturbations in the weights and biases allow one to give less error in
the functions.

-}

{-
SGD as defined here :
Given a
- seed value, seed
- some number of epochs, epochs
- a small (static) training rate, nu
- the number of examples in a batch, m
- a data source of input/output pairs, trainData
- a list of the dimensions of the vectors at intermediate layers, layerSizes
- an activation function and its derivative to be used over the entire network
    simga and sigma'
- an error derivative function, loss'

We produce a network of floating point valued weights, conforming to the
dimensions specified in the input, along with the activation function.

We outline how this algorithm recursively calls on lemmas, to break up the work.

The main "sgd" function, calls a helper function, "sgd'", which counts down from the
epochs to zero, updating the network over each data point over every epoch.

"singleEpochSGD", then breaks up the data into baches, as implemented by the
"makeBatches" function. "sgdOnBatches" then recursively trains each batch, calling
the "trainBatch" fucntion for every batch with the weights updated as regards the
previous batch.

The "trainBatch" function has to find the average weight and bias update
recommended over the batch, and does so by calculating the gradients using
"computeGradient" for every training example in the batch, and updateWeights

-}

sgd ::
  Seed             ->
  Epochs           ->
  TrainingRate     ->
  BatchSize        ->
  TrainData        ->
  LayerDimensions  ->
  ActivationFcn    ->
  ActivationFcn'   ->
  LossFunction'    ->
  Network
sgd seed epochs nu m trainData layerSizes sigma sigma' loss' =
  let initialNetwork     = generateRandomNetwork seed layerSizes
      (weights0,biases0) = unzip $ zeroNetwork layerSizes

      sgd' ::
        Epochs  ->
        Network ->
        Network
      sgd' 0 network = network
      sgd' epochs network =
        let trainedEpochN = singleEpochSGD trainData network
        in sgd' (epochs-1) trainedEpochN

      singleEpochSGD ::
        TrainData -> -- count this as an individual batch
        Network   ->
        Network
      -- singleEpochSGD seed nu m trainData layerSizes sigma' loss' weights0 biases0 n = n
      singleEpochSGD trainData network =
        let batches = makeBatches seed m trainData
        in sgdOnBatches batches network

      sgdOnBatches ::
        [TrainData] -> -- count this as an individual batch
        Network ->
        Network
      sgdOnBatches [] n = n
      sgdOnBatches (batch0:batches) n =
        let n' = trainBatch batch0 n
        in sgdOnBatches batches n'

      -- note theres a minor bug in that the batchsize scaling for the "modulo batch"
      trainBatch ::
        TrainData -> -- individual batch
        Network ->
        Network
      trainBatch batch net@(weightBiases,f) =
        let gradients = map (\b -> computeGradient b net loss' sigma') batch
            updatedWeights = updateWeights gradients weightBiases
        in (updatedWeights,f)

      updateWeights ::
        [(NetworkWeightGradients,NetworkBiasGradients)] ->
        NetworkWeights ->
        NetworkWeights
      updateWeights deltas currentWeights =
        let
          (weightGrads,biasGrads) = unzip deltas
          (weightsNow,biasNow)    = unzip currentWeights
          rate                    = nu / (fromIntegral m)
          sumBias                 = scalarMatMul rate $ foldr matMatSum biases0 biasGrads
          sumWeights              = scalarTensMul rate $ foldr tensTensSum weights0 weightGrads
          updateBias              = matMatDiff biasNow sumBias
          updateWeights           = tensTensDiff weightsNow sumWeights
        in zip updateWeights updateBias

  in sgd' epochs (initialNetwork,sigma)

--  BATCHES --

-- this is "data invariant"
makeBatches :: Seed -> BatchSize -> [a] -> [[a]]
makeBatches seed n [] = []
makeBatches seed n trainData =
  let seedN = mkStdGen seed
      shuffledData = fst $ shuffle' trainData seedN
  in spliceData shuffledData n

spliceData :: [a] -> Int -> [[a]]
spliceData [] _ = []
spliceData xs n =
  let (nxs,rest) = splitAt n xs
  in nxs : spliceData rest n

-- INITIAL NETWORK --

zeroNetwork :: [LayerSize] -> NetworkWeights
zeroNetwork sizes =
  let inputAndOutputDims = getPairs sizes
  in map zeroLayer inputAndOutputDims
  where
    zeroLayer :: (LayerSize,LayerSize) -> Layer
    zeroLayer (n,m) = (replicate m (replicate n 0), replicate m 0)
  -- generates a mxn matrix and m dimesnional vector

generateRandomNetwork ::
  Seed           ->
  [LayerSize]    -> -- include input and output layers, leght >=2
  NetworkWeights
generateRandomNetwork seed sizes =
  let inputAndOutputDims = getPairs sizes
      seeds = [seed..(seed + (length inputAndOutputDims))]
  in zipWith (\z -> \(x,y) -> initialLayerWeights y x z) seeds inputAndOutputDims

-- m : input dimension
-- n : output dimension
-- or n by m matrix
initialLayerWeights :: Int -> Int -> Seed -> ([[Double]],[Double])
initialLayerWeights n m seed =
  let combined = randListList seed (replicate n (m+1))
  in extractHeads combined

randListList :: Seed -> [Int] -> [[Double]]
randListList seed xs =
  let randStream = map (/20) $ randoms (mkStdGen seed)
  in randomMatrix xs randStream
  where
    randomMatrix :: [Int] -> [Double] -> [[Double]]
    randomMatrix [] xs = []
    randomMatrix (length:lengths) str =
      let (layerOne,rest) = splitAt length str
      in layerOne : randomMatrix lengths rest

-- anticipating seperation of matrix weights and biases
extractHeads :: [[x]] -> ([[x]],[x])
extractHeads xs =
  let (bs,ws) = unzip $ map (splitAt 1) xs
  in (ws,concat bs)

--BoilerPlate
getWeightedInputs :: [(WeightedInput,Activation)] -> [WeightedInput]
getWeightedInputs = map fst

getActivations :: [(WeightedInput,Activation)] -> [Activation]
getActivations = map snd
