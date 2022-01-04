module Tree where

-- import
import Data.Functor.Identity
import Control.Monad.State.Lazy


data Tree a = Leaf a | Branch (Tree a) (Tree a)
  deriving (Eq,Show)

t = Branch (Leaf "a")
  (Branch (Leaf "b") (Leaf "c"))


-- >>> runState (nlabel t) [3,51]
-- (Branch (Leaf 3) (Branch (Leaf 51) (Leaf *** Exception: /tmp/dantedldfZF.hs:23:18-35: Non-exhaustive patterns in lambda

--so how to do this without runtime exceptions?

tock :: State [Int] Int
tock = state (\(x:xs) -> (x,xs)) --(s,s+1)

nlabel :: Tree a -> State [Int] (Tree Int)
nlabel (Leaf x) =
  do n <- state (\(x:xs)  -> (x,xs)) 
     return (Leaf n)
nlabel (Branch t1 t2) =
  do n1 <- nlabel t1
     n2 <- nlabel t2
     return (Branch n1 n2)

-- >>> runState (mlabel t) 3
-- (Branch (Leaf 3) (Branch (Leaf 4) (Leaf 5)),6)

tick :: State Int Int
tick = state (\s -> (s,s+1)) --(s,s+1)

mlabel :: Tree a -> State Int (Tree Int)
mlabel (Leaf x) =
  do n <- tick
     return (Leaf n)
mlabel (Branch t1 t2) =
  do n1 <- mlabel t1
     n2 <- mlabel t2
     return (Branch n1 n2)

-- -- number :: Tree a -> Integer -> Tree (Integer)
-- number (Leaf a) =
--   tick >>= \s ->
--   return (Leaf s)
-- number (Branch l r) =
--   number l >>= \l' ->
--   number r >>= \r' ->
--   return (Branch l' r')
