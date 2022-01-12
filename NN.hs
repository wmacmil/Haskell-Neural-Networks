module NN where

import Shuffle
-- import Data.List
import System.Random
import Data.Tuple --swap

-- import Control.Monad.Random

-- the types give us a framework to refactor the code later, but keep at least the signature

-- module Linear algebra
type Vector a = [a]
type Matrix a = [Vector a]
type Tensor3 a = [Matrix a]

getPairs :: [a] -> [(a,a)]
getPairs [] = []
getPairs [x] = []
getPairs (x:y:xs) = (x,y) : getPairs (y:xs)

mat2 :: Matrix Double
mat2 = [[1.0,2],[3,4]]

-- nn's are typically done over R^n
--we'll only deal with (idealized) vector spaces over the reals
type R = Double
type Rn = Vector Double
type Rnm = Matrix Double
type Rnml = Tensor3 Double

dot :: Rn -> Rn -> R
dot x y = foldr (+) 0 (zipWith (*) x y)

vecVecSum :: Rn -> Rn -> Rn
vecVecSum x y = zipWith (+) x y

vecVecDiff :: Rn -> Rn -> Rn
vecVecDiff x y = zipWith (-) x y

--hamamard product
vecVecMul :: Rn -> Rn -> Rn
vecVecMul x y = zipWith (*) x y

--"canonical" notion of vector length
l2Norm :: Rn -> R
l2Norm xs = sqrt (dot xs xs)

--could get the above by partially evaluating to two
-- also the first argument should be a nat
lpNorm :: R -> Rn -> R
lpNorm p xs = (foldr (+) 0 (map (**p) xs))  ** (1 / p)

-- note that because *real* matrices are dependently typed by their indices,
-- this type does not give compile time gurantees
matVectProd :: Rnm -> Rn -> Rn
matVectProd m v = map (\x -> dot x v) m

-- >>> matMatDiff [[1,2],[3,4]] [[3,4],[4,3]]
-- [[-2.0,-2.0],[-1.0,1.0]]
matMatDiff :: Rnm -> Rnm -> Rnm
matMatDiff xss yss = zipWith vecVecDiff xss yss

matMatSum :: Rnm -> Rnm -> Rnm
matMatSum xss yss = zipWith vecVecSum xss yss

scalarMatMul :: R -> Rnm -> Rnm
scalarMatMul c yss = map (map (*c)) yss

scalarTensMul :: R -> Rnml -> Rnml
scalarTensMul c yss = map (map (map (*c))) yss

tensTensSum :: Rnml -> Rnml -> Rnml
tensTensSum = zipWith matMatSum

tensTensDiff :: Rnml -> Rnml -> Rnml
tensTensDiff = zipWith matMatDiff

affineTransform :: Rnm -> Rn -> Rn -> Rn
affineTransform  m x b = vecVecSum (matVectProd m x) b

vec2Mat :: Rn -> Rnm
vec2Mat = map (:[])

-- >>> map (:[]) [1..2]
-- [[1],[2]]

-- >>> matMatProd [[1..3],[4..6],[7..9]] [[0,1,0],[1,0,0],[0,0,1]]
-- [[2.0,1.0,3.0],[5.0,4.0,6.0],[8.0,7.0,9.0]]
-- >>> matMatProd [[0],[1]] [[2],[3]]
-- [[0.0,0.0],[2.0,3.0]]
matMatProd :: Rnm -> Rnm -> Rnm
matMatProd m1 m2 = map (\x -> matVectProd m2 x) m1


-- will give runtime error for non-matrix valued argument
-- >>> transpose [[1..5],[2..6]]
-- [[1.0,2.0],[2.0,3.0],[3.0,4.0],[4.0,5.0],[5.0,6.0]]
transpose :: Rnm -> Rnm
transpose [] = []
transpose ([] : ys) = []
transpose xss@((y : ys) : xs) = (map head xss) : transpose (map tail xss)

-- convetion to use ' to actually mean derivative
-- no runtmime when using type
-- distinguish between weights in and weights out
type ActivationFcn          = R -> R
type NodeBias               = R
type NeuronWeights          = Rn
type Neuron                 = (NeuronWeights, NodeBias) -- perceptron
type LayerBias              = [NodeBias]
type LayerSize              = Int
type NetworkDepth           = Int
type LayerWeights           = [NeuronWeights]
type Layer                  = (LayerWeights, LayerBias)
type NetworkWeights         = [Layer]
type Network                = (NetworkWeights,ActivationFcn) -- can be a list of activations
type LayerWeightGradients   = Rnm -- dC/dW_ij^l
type LayerBiasGradients     = Rn -- dC/dW_ij^l
type NetworkWeightGradients = [LayerWeightGradients]
type NetworkBiasGradients   = [LayerBiasGradients]
-- type Depth               = Int
type Input                  = Rn
type WeightedInput          = Rn --z
type Activation             = Rn --a
type Output                 = Rn
type ActivationFcn'         = Output -> Rn -- should make this consistent and de-vectorize
type Inputs                 = [Input]
type TrainData              = [(Input,Output)]
type TestData               = [(Input,Output)]
type Data                   = (TrainData,TestData)
type LossFunction           = Rn -> Rn -> R
type LossFunction'          = Rn -> Rn -> Rn
type TrainingRate           = R
type GradientDescent        = (TrainData,Network,LossFunction,TrainingRate)
type LayerError             = Rn

weights :: NetworkWeights -> [LayerWeights]
weights = map fst

-- >>> map sigmoid ((-5) : 0 : [5])
-- [6.692851e-3,0.5,0.9933072]
sigmoid :: ActivationFcn
sigmoid x = 1 / (1 + exp (- x))

--overloaded in the textbooks, underscore indicates vector
sigma_ = map sigmoid

epsilon0 :: TrainingRate
epsilon0 = 0.01

meanSquaredError':: LossFunction'
meanSquaredError' target predicted = vecVecDiff target predicted

-- could generalize with a buildNetwork Function
sigmaNetwork :: NetworkWeights -> Network
sigmaNetwork n = (n,sigmoid)

--we'll start out assuming a sigmoid activation
--i.e. partially evaluate these below to include a sigmoid
feedForwardSigmoid is (n,a) = feedForward is (n,sigmoid)


sigmoid' :: ActivationFcn'
sigmoid' output = vecVecMul output (vecVecDiff n1s output)
  where
    n1s = replicate (length output) 1

-- activation, a^l, and weighted input, z^l, at layer l
singleLayer :: Input -> Layer -> ActivationFcn -> (WeightedInput,Activation)
singleLayer ins (weights,bias) activate =
  let weighted = (affineTransform weights ins bias)
      activation = map activate weighted -- refactor renaming 'map activate'
  in (weighted,activation)

-- helper function
getWeightedInputs :: [(WeightedInput,Activation)] -> [WeightedInput]
getWeightedInputs = map fst

getActivations :: [(WeightedInput,Activation)] -> [Activation]
getActivations = map snd

-- everything here should be classified as compute gradient

-- base case might be wrong, check this
feedForward :: Input -> Network -> [(WeightedInput,Activation)]
feedForward ins ([],activate) = [(ins,ins)]
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

generateGradients :: [LayerError] -> [Activation] -> (NetworkWeightGradients,NetworkBiasGradients)
generateGradients delta_s a_s = (zipWith weightupdates delta_s a_s,delta_s )
  where
    reverse_a_s = reverse a_s
    weightupdates delta a = matMatProd (vec2Mat delta) (vec2Mat a)

computeGradient ::
  (Input,Output) ->
  -- Input ->
  -- Output ->
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


updateWeights ::
  TrainingRate ->
  BatchSize ->
  [(NetworkWeightGradients,NetworkBiasGradients)] ->
  NetworkWeights ->
  [LayerWeights] -> -- zeros
  [LayerBias] ->    -- zeros
  NetworkWeights
updateWeights nu m deltas currentWeights weights0 biases0 =
  let
    (weightGrads,biasGrads) = unzip deltas
    (weightsNow,biasNow) = unzip currentWeights
    rate = (nu / (fromIntegral m))
    sumBias = scalarMatMul rate $ foldr matMatSum biases0 biasGrads
    sumWeights = scalarTensMul rate $ foldr tensTensSum weights0 weightGrads
    updateBias = matMatDiff biasNow sumBias
    updateWeights = tensTensDiff weightsNow sumWeights
  in zip updateWeights updateBias

-- initialNetowork = (generateRandomNetwork seed layerSizes, sigma)

-- batchGradients = map (\xy -> computeGradient

  -- Network ->
  -- LossFunction'  ->
  -- ActivationFcn' ->

-- computeGradient ::
--   (Input,Output) ->
--   -- Input ->
--   -- Output ->
--   Network ->
--   LossFunction'  ->
--   ActivationFcn' ->
--   (NetworkWeightGradients,NetworkBiasGradients)

-- updateWeights ::
--   TrainingRate ->
--   BatchSize ->
--   [(NetworkWeightGradients,NetworkBiasGradients)] ->
--   NetworkWeights ->
--   [LayerWeights] -> -- zeros
--   [LayerBias] ->    -- zeros
--   NetworkWeights

trainBatch ::
  TrainingRate ->
  BatchSize ->
  TrainData -> -- individual batch
  Network ->
  LossFunction'  ->
  ActivationFcn' ->
  [LayerWeights] -> -- zeros
  [LayerBias] ->    -- zeros
  Network
trainBatch nu m batch net@(weightBiases,f) loss' sigma' weights0 biases0 =
  let gradients = map (\b -> computeGradient b net loss' sigma') batch
      updatedWeights = updateWeights nu m gradients weightBiases weights0 biases0
  in (updatedWeights,f)

sgd ::
  Seed ->
  Int -> -- number of epochs
  TrainingRate ->
  BatchSize -> --needed to
  TrainData -> -- count this as an individual batch
  [LayerSize] ->
  ActivationFcn ->
  ActivationFcn' ->
  LossFunction'  ->
  Network
sgd seed epochs nu m trainData layerSizes sigma sigma' loss' =
  let initialNetwork = generateRandomNetwork seed layerSizes
      (weights0,biases0) = unzip $ zeroNetwork layerSizes
  in sgdhelper seed epochs nu m trainData layerSizes sigma' loss' weights0 biases0 (initialNetwork,sigma)

sgdhelper ::
  Seed ->
  Int ->
  TrainingRate ->
  BatchSize -> --needed to
  TrainData -> -- count this as an individual batch
  [LayerSize] ->
  ActivationFcn' ->
  LossFunction'  ->
  [LayerWeights] -> -- zeros
  [LayerBias] ->    -- zeros
  Network ->
  Network
sgdhelper seed 0 nu  m trainData layerSizes sigma' loss' weights0 bias0 network = network
sgdhelper seed epochs nu  m trainData layerSizes sigma' loss' weights0 bias0 net =
  let foo = singleEpochSGD seed nu m trainData layerSizes sigma' loss' weights0 bias0 net -- hh(net,sig)
  in sgdhelper seed (epochs-1) nu m trainData layerSizes sigma' loss' weights0 bias0 foo 

-- (batch0:batches) = makeBatches seed m trainData
singleEpochSGD ::
  Seed ->
  TrainingRate ->
  BatchSize -> --needed to
  TrainData -> -- count this as an individual batch
  [LayerSize] ->
  ActivationFcn' ->
  LossFunction'  ->
  [LayerWeights] -> -- zeros
  [LayerBias] ->    -- zeros
  Network ->
  Network
-- singleEpochSGD seed nu m trainData layerSizes sigma' loss' weights0 biases0 n = n
singleEpochSGD seed nu m trainData layerSizes sigma' loss' weights0 biases0 n =
  let batches = makeBatches seed m trainData
  in stochasticGradientDescentOnBatch seed nu m batches layerSizes sigma' loss' weights0 biases0 n

-- (batch0:batches) = makeBatches seed m trainData
stochasticGradientDescentOnBatch ::
  Seed ->
  TrainingRate ->
  BatchSize -> --needed to
  [TrainData] -> -- count this as an individual batch
  [LayerSize] ->
  ActivationFcn' ->
  LossFunction'  ->
  [LayerWeights] -> -- zeros
  [LayerBias] ->    -- zeros
  Network ->
  Network
stochasticGradientDescentOnBatch seed nu m [] layerSizes sigma' loss' weights0 biases0 n = n
stochasticGradientDescentOnBatch seed nu m (batch0:batches) layerSizes sigma' loss' weights0 biases0 n =
  let n' = trainBatch nu m batch0 n loss' sigma' weights0 biases0
  in stochasticGradientDescentOnBatch seed nu m batches layerSizes sigma' loss' weights0 biases0 n'

type Batches = [TrainData]
type BatchSize = Int
type Seed = Int

-- this is "data invariant"
makeBatches :: Seed -> BatchSize -> [a] -> [[a]]
makeBatches seed n [] = []
makeBatches seed n trainData =
  let seedN = mkStdGen seed
      shuffledData = fst $ shuffle' trainData seedN
  in spliceData shuffledData n

-- >>> zeroNetwork [2,3,4]
-- [([[0.0,0.0],[0.0,0.0],[0.0,0.0]],[0.0,0.0,0.0]),([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],[0.0,0.0,0.0,0.0])]

-- >>> zeroLayer (3,2)
-- ([[0.0,0.0,0.0],[0.0,0.0,0.0]],[0.0,0.0])

zeroLayer :: (LayerSize,LayerSize) -> Layer
zeroLayer (n,m) = (replicate m (replicate n 0), replicate m 0)

zeroNetwork :: [LayerSize] -> NetworkWeights
zeroNetwork sizes =
  let inputAndOutputDims = getPairs sizes
  in map zeroLayer inputAndOutputDims

generateRandomNetwork ::
  Seed           -> -- seed
  [LayerSize]    -> -- include input and output layers, leght >=2
  NetworkWeights
generateRandomNetwork seed sizes =
  let inputAndOutputDims = getPairs sizes
      seeds = [seed..(seed + (length inputAndOutputDims))]
  in zipWith (\z -> \(x,y) -> initialLayerWeights y x z) seeds inputAndOutputDims

-- NetworkDimensions = [LayerSizes]
-- dont do this until the end
-- just do everything with the batches up here

-- stochasticGradientDescent ::

-- stochasticGradientDescentOnEpoch ::
--   let batches = makeBatches seed batchSize trainData
-- batchTrainRate = nu / batchSize



-- Helper Functions Below --

-- m : input dimension
-- n : output dimension
-- or n by m matrix
-- could also include the choice of including a bias as input variable
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


extractHeads :: [[x]] -> ([[x]],[x])
extractHeads xs =
  let (bs,ws) = unzip $ map (splitAt 1) xs
  in (ws,concat bs)

spliceData :: [a] -> Int -> [[a]]
spliceData [] _ = []
spliceData xs n =
  let (nxs,rest) = splitAt n xs
  in nxs : spliceData rest n


-- >>> makeBatches 4 2 [1..9]
-- [[6,1],[2,5],[9,4],[3,8],[7]]

-- seems to be working
-- >>> let x = generateRandomNetwork 1 [2,3,4]
-- >>> (length $ snd $ head $ generateRandomNetwork 1 [2,3,4]) == 3
-- True
-- >>> (length $ fst $ head $ tail $ generateRandomNetwork 1 [2,3,4]) == 4
-- True
-- >>> (length $ head $ fst $ head $ tail $ generateRandomNetwork 1 [2,3,4]) == 3
-- True
-- >>> zeroNetwork [2,3,4]
-- [([[0.0,0.0],[0.0,0.0],[0.0,0.0]],[0.0,0.0,0.0]),([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],[0.0,0.0,0.0,0.0])]
-- >>> (length $ head $ fst $ head $ tail $ zeroNetwork [2,3,4]) == 3
-- True

-- >>> initialLayerWeights 2 3 10
-- ([[1.9011329192943332e-2,3.440097495171499e-2,2.2673567200129363e-2],[2.6473535110395907e-3,4.600034458036367e-2,3.091743248512812e-2]],[4.7055605478282456e-2,1.5564736642606918e-2])
-- >>> extractHeads [[1,2,3],[4,5,6]]
-- ([[2,3],[5,6]],[1,4])



-- feedForward :: Input -> Network -> [(WeightedInput,Activation)]

-- outputError  = computeOutputError
--                  target               --   Output         ->
--                  (head weightedInputs)--   WeightedInput  ->
--                  (head activations)   --   Activation     ->
--                  loss'                --   LossFunction'  ->
--                  sigma'               --   ActivationFcn' ->
--                                            LayerError




-- TODO :
--   * Momenumify
--   * Generalize
--   * Rector to have better random geneartion possibility
--   * give overloaded ways of giving a backprop algorithm
--     + i.e. make functions that give the network implicitly, vs just some
--       layer parameter (d = depth, bread_i and acitivation_i for i in depth),
--       more generality

-- type Bactivation = Activation --for backprop
-- -- can change this to adjust for semantic
-- -- not sure if this is relevant, because we actually just compute the values
-- dSigmoid :: Bactivation
-- dSigmoid x = sigmoid x * (1 - sigmoid x)

-- -- also, this is really only needed when we're testing the data, not training
-- -- because the outputs out each layer are used during backprop
-- -- one should probably refactor to just include the activation at the
-- -- network level
-- -- many layers compose from single layers
-- manyLayer :: Input -> Network -> Output
-- manyLayer i ([],activate) = i
-- manyLayer i ((l:ls),activate) =
--   manyLayer (singleLayer i l activate) (ls,activate)


-- -- >>> map (relu . (*) (-1)) ([1..10] ++ [-1,-2])
-- -- [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0]
-- relu :: ActivationFcn
-- relu x = max 0 x

-- sgn :: R -> Bool
-- sgn x = if x > 0 then True else False

-- --could define neutral neuron with sgn
-- perceptron :: Neuron -> Input -> Bool
-- perceptron (w,b) i = sgn (dot w i)
