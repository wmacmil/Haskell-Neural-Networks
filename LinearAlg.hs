module LinearAlg where

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

