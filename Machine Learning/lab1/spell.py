import sys
import math

def read_words(filename):
	f = open(filename)
	counters = {}
	total = 0
	
	for line in f:
		for word in line.lower().split():
			if word.isalpha():
				if word in counters:
					counters[word] += 1
				else:
					counters[word] = 1
				total += 1
	f.close()
	
	priors = {}
	
	for word in counters:
		priors[word] = counters[word] / total
	
	return priors

DICTIONARY = read_words("big.txt")

def binomial_coeff (n, k):
	return math.factorial(n) // (math.factorial(k)) * (math.factorial(n-k))

def err_prob(e, n, q):
	return binomial_coeff(n, e) * (q ** e) * ((1-q) ** (n-e))

def edit1(word):
	n = len(word)
	letters = "abcdefghijklmnopqrstuvwxyz"
	variations = set()
	
	for i in range(n+1):
		for l in letters:
			newword = word[:i] + l + word[i:]
			variations.add(newword)
			
			if i < n:
				newword = word[:i] + l + word[(i+1):]
				variations.add(newword)
			
		if i < n:
			newword = word[:i] + word[(i+1):]
			variations.add(newword)
		
		if i+1 < n:
			newword = word[:i] + word[i+1] + word[i] + word[(i+1):]
			variations.add(newword)
	
	return variations

def edit_k(word, k):
	variations = {word}
	
	for _ in range(k):
		newvars = set()
		for v in variations:
			newvars |= edit1(v)
		variations = newvars
	
	return variations

def correct_word(word, maxerr, q, top):
	candidates = []

	for e in range(maxerr + 1):
		variations = edit_k(word, e)
		for c in variations:
			if c in DICTIONARY:
				p_c = DICTIONARY[c]
				p_wc = err_prob(e, len(c), q)
				score = p_wc * p_c
				candidates.append((score, c))
				
	candidates.sort(reverse=True)
	return candidates[:top]

word = sys.argv[1]
candidates = correct_word(word, 2, 0.01, 5)

for score, c in candidates:
	print(c)




