module Verification where

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

-- >>> zeroNetwork [2,3,4]
-- [([[0.0,0.0],[0.0,0.0],[0.0,0.0]],[0.0,0.0,0.0]),([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],[0.0,0.0,0.0,0.0])]

-- >>> zeroLayer (3,2)
-- ([[0.0,0.0,0.0],[0.0,0.0,0.0]],[0.0,0.0])

