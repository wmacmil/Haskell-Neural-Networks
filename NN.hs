module NN where

import Shuffle
-- import Data.List
import System.Random
-- import Control.Monad.Random

-- the types give us a framework to refactor the code later, but keep at least the signature

-- module Linear algebra
type Vector a = [a]
type Matrix a = [Vector a]

mat2 :: Matrix Float
mat2 = [[1.0,2],[3,4]]

-- nn's are typically done over R^n
--we'll only deal with (idealized) vector spaces over the reals
type R = Float
type Rn = Vector Float
type Rnm = Matrix Float

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

-- zipwith (+) [1..3] [2..4]
generateGradients :: [LayerError] -> [Activation] -> (NetworkWeightGradients,NetworkBiasGradients)
generateGradients delta_s a_s = (zipWith weightupdates delta_s a_s,delta_s )
  where
    reverse_a_s = reverse a_s
    weightupdates delta a = matMatProd (vec2Mat delta) (vec2Mat a)

computeGradient ::
  Input ->
  Output ->
  Network ->
  LossFunction'  ->
  ActivationFcn' ->
  (NetworkWeightGradients,NetworkBiasGradients)
computeGradient x target n@(weightsAndBiases,_) loss' sigma' =
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


type Batches = [TrainData]
type BatchSize = Int

-- seed : _
seed3 = mkStdGen 3

-- >>> splitAt 10 [1..8]
-- ([1,2,3,4,5,6,7,8],[])
-- >>> makeBatches 10 [1..91]
-- [[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29,30],[31,32,33,34,35,36,37,38,39,40],[41,42,43,44,45,46,47,48,49,50],[51,52,53,54,55,56,57,58,59,60],[61,62,63,64,65,66,67,68,69,70],[71,72,73,74,75,76,77,78,79,80],[81,82,83,84,85,86,87,88,89,90],[91]]

-- this is "data invariant"
makeBatches :: BatchSize -> [a] -> [[a]]
makeBatches n [] = []
makeBatches n trainData =
  let shuffledData = fst $ shuffle' trainData seed3
      (xss,rest) = splitAt n shuffledData
  in xss : (makeBatches n rest)

-- type LayerSize              = Int
-- type NetworkDepth           = Int

-- generateRandomNetwork ::
--   Int            -> -- seed
--   [LayerSize]    -> -- include input and output layers, leght >=2
--   -- NetworkDepth   ->
--   NetworkWeights
-- generateRandomNetwork seed (size:size0:sizes) = _
-- generateRandomNetwork seed (size:size0:sizes) = _
--   -- where
--   --   randomizeLayer = _
--   --   layerWidth = _

-- for generating random initial weights

-- >>> randomList 3 6
-- [2.7870901893892748e-2,2.81191257317836e-3,2.1600613796659333e-2]
-- >>> fst $ splitAt 10 $ snd $ splitAt 10 (randStream 3)
-- [0.6691526198493818,0.25545946031582445,0.8445748363433679,0.5286878094875974,0.7322882841255793,0.7350424276133447,0.4721000032845192,8.065993815509054e-2,0.23423796590225265,0.5642031099938646]

-- [2.7870901893892748e-2,2.81191257317836e-3,2.1600613796659333e-2]

randStream seed = randoms (mkStdGen seed) :: [Double]

-- something :: Int

-- something :: [Int] -> [Double] -> [[Double]]

-- >>> splitAt 3 (randStream 4)

-- >>> randStream 100 4

-- >>> splitAt 3 [1,2]
-- ([1,2],[])

rand4100 = randStream 4

-- >>> map length $ randomMatrix [12,10,24] rand4100
-- >>> randomMatrix [1..10] (randStream 5)
--

randomMatrix :: [Int] -> [Double] -> [[Double]]
randomMatrix [] xs = []
randomMatrix (length:lengths) str = -- []
  let (layerOne,rest) = splitAt length str
  in layerOne : randomMatrix lengths rest


  -- let (layerOne,rest) = splitAt length $ randStream
  -- in layerOne : (randomMatrix
  -- where
  --   randStream = randoms (mkStdGen seed) :: [Double]

  -- let shuffledData = fst $ shuffle' trainData seed3
  --     (xss,rest) = splitAt n shuffledData
  -- in xss : (makeBatches n rest)

randomList :: Int -> Int -> [Double]
randomList length seed = map (/20) $ take length $ randStream
  where
    randStream = randoms (mkStdGen seed) :: [Double]

-- type Layer                  = (LayerWeights, LayerBias)
-- type NetworkWeights         = [Layer]
-- type Network                = (NetworkWeights,ActivationFcn) -- can be a list of activations

-- >>> foldr (take 10) _ [1..100]
-- <interactive>:873:9-15: error:
--     • Couldn't match type ‘[a0]’ with ‘[a] -> [a]’
--       Expected type: [a0] -> [a] -> [a]
--         Actual type: [a0] -> [a0]
--     • Possible cause: ‘take’ is applied to too many arguments
--       In the first argument of ‘foldr’, namely ‘(take 10)’
--       In the expression: foldr (take 10) [] [1 .. 100]
--       In an equation for ‘it’: it = foldr (take 10) [] [1 .. 100]
--     • Relevant bindings include
--         it :: [a] (bound at <interactive>:873:2)

-- >>> :t take
-- take :: Int -> [a] -> [a]

-- >>> fst $ shuffle' [1..10] seed
-- [4,5,3,10,8,2,7,6,9,1]

-- -- idea : add batches to buckets
-- choose a random permutaion of the n numbers, (1..n) (i,...,1,..n,..j)
-- and then speficy which bucket to add it to
-- genBatches :: BatchSize -> TrainData -> Batches
-- genBatches size xss@(x:xs) = _
--   where
--     numBatches = 1 + div (length xss) size


-- Now to actually run *stochastic* gradient desent, we need to
-- (i) Create random batches of input data, paramaterized by a nat
-- (ii) Run the batches via some number of epochs
-- (iii) Initiate random weights to begin with

-- feedForward :: Input -> Network -> [(WeightedInput,Activation)]

-- outputError  = computeOutputError
--                  target               --   Output         ->
--                  (head weightedInputs)--   WeightedInput  ->
--                  (head activations)   --   Activation     ->
--                  loss'                --   LossFunction'  ->
--                  sigma'               --   ActivationFcn' ->
--                                            LayerError

-- backPropError :
--   LayerError      -> -- initial error
--   -- Depth           ->    -- could decrease from depth of the network, implicit
--   ActivationFcn'  ->
--   NetworkWeights  ->
--   [WeightedInput] ->
--   [LayerError]
-- generateGradients :: [LayerError] -> [Activation] -> (NetworkWeightGradients,NetworkBiasGradients)
-- generateGradients delta_s a_s = (zipWith weightupdates delta_s a_s,delta_s )


-- to put it all together


-- output :: LayerError -> Activation -> (LayerWeightGradients,LayerBiasGradients)
-- output (x:xs) (a,as) =

-- -- errors and weighted inputs shou
-- backPropError ::
--   Depth          ->
--   ActivationFcn' ->
--   NetworkWeights ->
--   [WeightedInput]  ->
--   [LayerError]
-- backPropError 0 ws sigma' (z:zs) = [] -- outputError
-- backPropError d sigma' ((ws,bs):wss) (z:zs) = (helper) : (backPropError (d-1) sigma' wss zs)
--   where
--     helper :: LayerError
--       -- Depth          ->
--       -- ActivationFcn' ->
--       -- NetworkWeights ->
--       -- WeightedInput  ->
--       -- LayerError
--     helper =
--       let delta_lPlus1 = helper (d-1) sigma' wss zs
--           delta_l = vecVecMul (matVectProd ws delta_lPlus1) (sigma' z) --transpose needed
--       in _



-- trainOnExample :: (Input,Output) -> Network -> LossFunction -> Network
-- trainOnExample (i,output) n = _
--   where
--     localOutput = feedForward i n
--     deltaOut = vecVecDiff output localOutput

  -- manyLayer ::
-- manyLayer i ([],activate) = i
-- manyLayer i ((l:ls),activate) =
--   manyLayer (singleLayer i l activate) (ls,activate)


-- meanSquaredError :: LossFunction
-- meanSquaredError yObserved yPrededicted =

-- TODO :
--   * Batchify
--   * Momenumify
--   * Generalize
--   * Rector to have better random geneartion possibility
--   * give overloaded ways of giving a backprop algorithm
--     + i.e. make functions that give the network implicitly, vs just some
--       layer parameter (d = depth, bread_i and acitivation_i for i in depth),
--       more generality

-- backpropogation ::
--   TrainData ->
--   TrainingRate ->
--   -- Activation ->
--   LossFunction ->
--   Network -> -- randomized initial weights
--   Network -> -- new network with trained output weights
-- backpropogation (ins,targetOuts) nu loss (weights,f) = (updatedWeights,f)
--   where
--     updatedWeights :: NetworkWeights
--     updatedWeights = _
--     outputs = map manyLayer ins


-- Example : 2 layer netork, each layer having 5 weights,



-- singleHiddenBackProp ::
--   TrainData ->
--   TrainingRate ->
--   LossFunction ->
--   Network -> -- randomized initial weights
--   Network -> -- new network with trained output weights
-- backpropogation (ins,targetOuts) nu loss (weights,f) = (updatedWeights,f)
--   where
--     updatedWeights :: NetworkWeights
--     updatedWeights = _
--     outputs = map manyLayer ins


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

-- meanSqError :: ErrorFunction
-- meanSqError =

-- -- >>> map (relu . (*) (-1)) ([1..10] ++ [-1,-2])
-- -- [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0]
-- relu :: ActivationFcn
-- relu x = max 0 x


-- initializeLayerWeights :: Layer
-- initializeNetwork :: Activation -> Network

-- sgn :: R -> Bool
-- sgn x = if x > 0 then True else False

-- --could define neutral neuron with sgn
-- perceptron :: Neuron -> Input -> Bool
-- perceptron (w,b) i = sgn (dot w i)


-- type Layer = ([NeuronWeights], LayerBias, Activation)
-- type Network = [Layer]
-- test with quickcheck
-- import Test.QuickCheck
-- import Util hiding (replicate)
-- vectors as lists, first order approximation
-- refactor with Data.Vector zipWithG
-- refactor with linear, or hmatrix
