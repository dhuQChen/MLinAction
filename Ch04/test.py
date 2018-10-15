import re

mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
# print(mySent.split())
regEx = re.compile('\\W+')
# listOfTokens = regEx.split(mySent)
# listOfTokens = [tok.lower() for tok in listOfTokens if len(tok) >= 1]
# print(listOfTokens)
emailText = open('email/ham/6.txt').read()
listOfTokens = regEx.split(emailText)
listOfTokens = [tok.lower() for tok in listOfTokens if len(tok) >= 3]
print(listOfTokens)