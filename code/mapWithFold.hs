


map' :: (a -> b) -> [a] -> [b]
map' f [] = []
map' f (x : xs) = f x : (map' f xs)

-- >>> :t foldr
-- foldr :: Foldable t => (a -> b -> b) -> b -> t a -> b


-- if b := a -> a
foldr' :: (a -> b -> b) -> b -> [a] -> b
foldr' fab b [] = b
foldr' fab b (x:xs) = fab x $ foldr' fab b xs

mapWithState :: (a -> b) -> (a -> a) -> [a] -> [b] 
mapWithState f g [] = []
mapWithState f g (x:xs) = f x : mapWithState (f . g) g xs

-- >>> :t replicate
-- repeat :: a -> [a]

-- >>> mapWithState (+1) (+1) (replicate 3 1) -- [1..10] 
-- [2,3,4]

-- >>> :t (+1) 
-- (+1) :: Num a => a -> a
-- >>> :t ((+1) . (+1))
-- ((+1) . (+1)) :: Num c => c -> c

-- (+1) .  :: Int -> Int
  
-- a = foldr (\x y -> (+1) 

-- >>> map' (+1) [1..10]
-- [2,3,4,5,6,7,8,9,10,11]
-- >>> foldr' (+) 0 [1..10]
-- 55
-- >>> mapWithFold (+1) [1..10]
-- [2,3,4,5,6,7,8,9,10,11]

-- mapWithFold :: (a -> a) -> (a -> b) -> [a] -> [b]

-- mapWithFold :: (a -> b) -> [a] -> [b]
-- mapWithFold f xs = foldr' (\x y -> f x : y) [] xs
