module NN where

import System.Random

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
type Neuron                 = (NeuronWeights, NodeBias) -- perceptron
type LayerBias              = [NodeBias]
type LayerSize              = Int
type NeuronWeights          = Rn
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
      weightedInputs               = getWeightedInputs wghtdInputsAndActvns
      activations                  = getActivations wghtdInputsAndActvns
      outputError                  = computeOutputError
                                       target
                                       (head weightedInputs)
                                       (head activations)
                                       loss'
                                       sigma'
      transposeWeights             = map transpose (weights weightsAndBiases)
      networkErrors                = backPropError
                        outputError
                        sigma'
                        transposeWeights
                        weightedInputs
  in generateGradients networkErrors activations

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

-- for generating random initial weights
-- >>> randomList 3 3
-- [1.348290170669587e-2,2.8227453280748254e-2,3.719821842745461e-2]
randomList :: Int -> Int -> [Double]
randomList length seed = map (/20) $ take 3 $ randStream
  where
    randStream = randoms (mkStdGen seed) :: [Double]


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
