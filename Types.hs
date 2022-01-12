module Types where

import LinearAlg

-- convetion to use ' to actually mean derivative
-- no runtime overhead when using Type
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
type Batches = [TrainData]
type BatchSize = Int
type Seed = Int
