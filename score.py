import nltk
from nltk.translate.bleu_score import sentence_bleu

nltk.download('punkt')

def score_response(ground_truth, generated_response):

  # Tokenize both sentences
  ground_truth_tokens = nltk.word_tokenize(ground_truth.lower())
  generated_tokens = nltk.word_tokenize(generated_response.lower())

  # Calculate BLEU score
  bleu_score = sentence_bleu([ground_truth_tokens], generated_tokens)
  
  # Calculate percentage of words that match
  match_count = len(set(ground_truth_tokens).intersection(set(generated_tokens)))
  total_words = len(set(ground_truth_tokens).union(set(generated_tokens)))
  accuracy = match_count / total_words

  # Return overall score
  overall_score = 0.5 * bleu_score + 0.5 * accuracy
  
  return overall_score

# Sample usage
ground_truth = "Euler invented calculus in the 18th century" 
generated_response = "Euler developed calculus in the 1700s"

score = score_response(ground_truth, generated_response)
print(score) # 0.625