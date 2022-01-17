module BooleanFomulas where

import Data.Functor.Identity
import Control.Monad.State.Lazy

-- lets learn bool functions
-- want a notion of variable

data BoolFmla =
  And BoolFmla BoolFmla |
  Or BoolFmla BoolFmla  |
  Neg BoolFmla          |
  Atom Bool
  deriving Show

nand :: BoolFmla -> BoolFmla -> BoolFmla
nand b1 b2 = Neg (And b1 b2)

nandTT = nand (Atom True) (Atom True)

orNegAnd :: BoolFmla -> BoolFmla -> BoolFmla -> BoolFmla
orNegAnd b1 b2 b3 = Or b1 (Neg (And b2 b3))

sizeFmla :: BoolFmla -> Int
sizeFmla (And b1 b2) = (sizeFmla b1 + sizeFmla b2)
sizeFmla (Or b1 b2)  = (sizeFmla b1 + sizeFmla b2)
sizeFmla (Neg b)     = sizeFmla b
sizeFmla (Atom b)    = 1

evalFmla :: BoolFmla -> Bool
evalFmla (And b1 b2) = evalFmla b1 && evalFmla b2
evalFmla (Or b1 b2)  = evalFmla b1 || evalFmla b2
evalFmla (Neg b)     = not (evalFmla b)
evalFmla (Atom b)    = b

orNegAndI :: BoolFmla
orNegAndI = orNegAnd (Atom True) (Atom True) (Atom True)


-- tock :: State [Int] Int
-- tock = state (\(x:xs) -> (x,xs)) --(s,s+1)

-- nlabel :: Tree a -> State [Int] (Tree Int)
-- nlabel (Leaf x) =
--   do n <- state (\(x:xs)  -> (x,xs)) 
--      return (Leaf n)
-- nlabel (Branch t1 t2) =
--   do n1 <- nlabel t1
--      n2 <- nlabel t2
--      return (Branch n1 n2)


-- runtime error if size constraints arent enforced
-- replace with state monad
-- fill the nodes of the boolean formulas
fill :: BoolFmla -> [Bool] -> BoolFmla
fill = ((.).(.)) fst fillFmla
  where
    fillFmla :: BoolFmla -> [Bool] -> (BoolFmla,[Bool])
    fillFmla (Atom x) (b:bs) = ((Atom b),bs)
    fillFmla (Neg b) bs =
      let (fmla,bs') = fillFmla b bs
      in (Neg fmla,bs')
    fillFmla (And b1 b2) bs =
      let (fmla,bs') = fillFmla b1 bs
          (fmla',bs'') = fillFmla b2 bs'
      in (And fmla fmla',bs'')
    fillFmla (Or b1 b2) bs =
      let (fmla,bs') = fillFmla b1 bs
          (fmla',bs'') = fillFmla b2 bs'
      in (Or fmla fmla',bs'')


-- [[True,True,True],[True,True,False],[True,False,True],[True,False,False],[False,True,True],[False,True,False],[False,False,True],[False,False,False]]
genBools :: Int -> [[Bool]]
genBools 0 = [[]]
genBools n =
  let recCall = (genBools (n-1))
  in map ((:) True) recCall ++ map ((:) False) recCall

-- >>> findAnswers nandTT
-- ([[True,True],[True,False],[False,True],[False,False]],[False,True,True,True])
-- >>> findAnswers orNegAndI
-- ([[True,True,True],[True,True,False],[True,False,True],[True,False,False],[False,True,True],[False,True,False],[False,False,True],[False,False,False]],[True,True,True,True,False,True,True,True])
findAnswers :: BoolFmla -> ([[Bool]],[Bool])
findAnswers fmla = (bools ,evals)
  where
    size = sizeFmla fmla
    bools = genBools size
    evals = map evalFmla $ map (fill fmla) bools

-- can now generate data sets to train on

-- now i have a data set prepared

-- >>> powerSet [0,1]
-- [[0,1],[0],[1],[]]
-- power set is the number of subsets of a set
-- also, functions from a set to booleans
powerSet :: [a] -> [[a]]
powerSet [] = [[]]
powerSet [x] = [[x],[]]
powerSet (x : xs) = map ((:) x) (powerSet xs) ++ powerSet xs
